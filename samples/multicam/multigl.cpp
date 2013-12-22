/*!\file
 * \brief Real-time augmentation using OpenGL and shaders.
 *
 * This example first calibrates connected camera(s) and then uses OpenGL to
 * augment the calibration target with a teapot, shaded by the calibrated light
 * map.
 */
#include <iostream>
#include <vector>
#include <cv.h>
#include <highgui.h>

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <calib/camera.h>
#include "multigrab.h"

#ifdef HAVE_APPLE_OPENGL_FRAMEWORK
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

MultiGrab *multi=0;

int geom_calib_nb_homography;
CamCalibration *calib=0;
int current_cam = 0;
IplTexture *frameTexture=0;
bool frameOK=false;
int nbLightMeasures=0;
bool cacheLight=false;
bool dynamicLight=false;
bool sphereObject=false;

static void photo_start();
static void geom_calib_start(bool cache);

//! GLUT callback on window size change 
static void
reshape(int width, int height)
{
  GLfloat h = (GLfloat) height / (GLfloat) width;

  glViewport(0, 0, (GLint) width, (GLint) height);
  glutPostRedisplay();
}

//! Print a command line help and exit.
static void usage(const char *s) {
  cerr << "usage:\n" << s
       << "[-m <model image>] [-r]\n"
    "	-m	specifies model image\n"
    "	-r	do not load any data\n"
    "	-t	train a new classifier\n"
    "	-g	recompute geometric calibration\n"
    "	-l	rebuild irradiance map from scratch\n";
  exit(1);
}


/*!\brief Initialize everything
 *
 * - Parse the command line
 * - Initialize all the cameras
 * - Load or interactively build a model, with its classifier.
 * - Set the GLUT callbacks for geometric calibration or, if already done, for photometric calibration.
 */
static bool init( int argc, char** argv )
{
  bool redo_geom=false;
  bool redo_training=false;
  bool redo_lighting=false;

  char *modelFile = "model.bmp";
  // parse command line
  for (int i=1; i<argc; i++) {
    if (strcmp(argv[i], "-m") ==0) {
      if (i==argc-1) usage(argv[0]);
      modelFile = argv[i+1];
      i++;
    } else if (strcmp(argv[i], "-r")==0) {
      redo_geom=redo_training=redo_lighting=true;
    } else if (strcmp(argv[i], "-g")==0) {
      redo_geom=redo_lighting=true;
    } else if (strcmp(argv[i], "-l")==0) {
      redo_lighting=true;
    } else if (strcmp(argv[i], "-t")==0) {
      redo_training=true;
    } else if (argv[i][0]=='-') {
      usage(argv[0]);
    } 
  }

  cacheLight = !redo_lighting;

  multi = new MultiGrab(modelFile);

  if( multi->init(!redo_training) ==0 )
    {
      cerr <<"Initialization error.\n";
      return false;
    }

  geom_calib_start(!redo_geom);

  return true;
}

/*! The keyboard callback: reacts to '+' and '-' to change the viewed cam, 'q' exits.
 * 'o' switches between a teapot and a sphere
 * 'd' turns on/off the dynamic lightmap update.
 * 'f' goes fullscreen.
 */
static void keyboard(unsigned char c, int x, int y)
{
  switch (c) {
  case 'n' :
  case '+' : if (current_cam < multi->cams.size()-1) 
      current_cam++;
    break;
  case 'p':
  case '-': if (current_cam >= 1) 
      current_cam--;
    break;
  case 'q': exit(0); break;
  case 'd': dynamicLight = !dynamicLight; break;
  case 'o': sphereObject = !sphereObject; break;
  case 'f': glutFullScreen(); break;
  }
  glutPostRedisplay();
}

static void emptyWindow() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

int main(int argc, char *argv[])
{

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);

  glutCreateWindow("Multi-Cam Teapot augmentation");
  glutDisplayFunc(emptyWindow);

  if (!init(argc,argv)) return -1;

  cvDestroyAllWindows();

  glutKeyboardFunc(keyboard);
  glutMainLoop();
  return 0;             /* ANSI C requires main to return int. */
}

//!\brief  Draw a frame contained in an IplTexture object on an OpenGL viewport.
static bool drawBackground(IplTexture *tex)
{
  if (!tex || !tex->getIm()) return false;

  IplImage *im = tex->getIm();
  int w = im->width-1;
  int h = im->height-1;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glDisable(GL_BLEND);
  glDisable(GL_DEPTH_TEST);

  tex->loadTexture();

  glBegin(GL_QUADS);
  glColor4f(1,1,1,1);

  glTexCoord2f(tex->u(0), tex->v(0));
  glVertex2f(-1, 1);

  glTexCoord2f(tex->u(w), tex->v(0));
  glVertex2f(1, 1);

  glTexCoord2f(tex->u(w), tex->v(h));
  glVertex2f(1, -1);

  glTexCoord2f(tex->u(0), tex->v(h));
  glVertex2f(-1, -1);
  glEnd();

  tex->disableTexture();

  return true;
}

/*! \brief A draw callback during camera calibration
 *
 * GLUT calls that function during camera calibration when repainting the
 * window is required.
 * During geometric calibration, no 3D is known: we just plot 2d points
 * where some feature points have been recognized.
 */
static void geom_calib_draw(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glDisable(GL_LIGHTING);
  drawBackground(frameTexture);

  if (!multi) return;

  IplImage *im = multi->cams[current_cam]->frame;
  planar_object_recognizer &detector(multi->cams[current_cam]->detector);
  if (!im) return;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, im->width-1, im->height-1, 0, -1, 1);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glDisable(GL_BLEND);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);

  if (detector.object_is_detected) {

    glPointSize(2);
    glBegin(GL_POINTS);
    glColor4f(0,1,0,1);
    for (int i=0; i<detector.match_number; ++i) {

      image_object_point_match * match = detector.matches+i;
      if (match->inlier) {
	int s = (int)(match->image_point->scale);
	float x=PyrImage::convCoordf(match->image_point->u, s, 0);
	float y=PyrImage::convCoordf(match->image_point->v, s, 0);
	glVertex2f(x,y);
      }
    }
    glEnd();
  }

  glutSwapBuffers();
}

/*!\brief Called when geometric calibration ends. It makes
 * sure that the CamAugmentation object is ready to work.
 */
static void geom_calib_end()
{

  if (!multi->model.augm.LoadOptimalStructureFromFile("camera_c.txt", "camera_r_t.txt")) 
    {
      cout << "failed to load camera calibration.\n";
      exit(-1);
    }
  glutIdleFunc(0);
  //glutDisplayFunc(0);
  delete calib;
  calib=0;
}

/*! Called by GLUT during geometric calibration when there's nothing else to do.
 * This function grab frames from camera(s), run the 2D detection on every image,
 * and keep the result in memory for calibration. When enough homographies have
 * been detected, it tries to actually calibrate the cameras.
 */
static void geom_calib_idle(void)
{
  // acquire images
  multi->grabFrames();

  // detect the calibration object in every image
  // (this loop could be paralelized)
  int nbdet=0;
  for (int i=0; i<multi->cams.size(); ++i) {
    if (multi->cams[i]->detect()) nbdet++;
  }

  if(!frameTexture) frameTexture = new IplTexture;
  frameTexture->setImage(multi->cams[current_cam]->frame);

  if (nbdet>0) {
    for (int i=0; i<multi->cams.size(); ++i) {
      if (multi->cams[i]->detector.object_is_detected) {
	add_detected_homography(i, multi->cams[i]->detector, *calib);
      } else {
	calib->AddHomography(i);
      }
    }
    geom_calib_nb_homography++;
  }

  if (geom_calib_nb_homography>=150) {
    if (calib->Calibrate(
			 50, // max hom
			 (multi->cams.size() > 1 ? 1:2),   // padding or random
			 (multi->cams.size() > 1 ? 0:3),
			 1,   // padding ratio 1/2
			 0,
			 0,
			 0.0078125,	//alpha
			 0.9,		//beta
			 0.001953125,//gamma
			 10,	  // iter
			 0.05, //eps
			 3   //postfilter eps
			 )) 
      {
	calib->PrintOptimizedResultsToFile1();
	geom_calib_end();
	photo_start();
	return;
      }
  }

  glutPostRedisplay();
}

/*!\brief Start geometric calibration. If the calibration can be loaded from disk,
 * continue directly with photometric calibration.
 */
static void geom_calib_start(bool cache)
{
  if (cache && multi->model.augm.LoadOptimalStructureFromFile("camera_c.txt", "camera_r_t.txt")) {
    photo_start();
    return;
  }

  // construct a CamCalibration object and register all the cameras
  calib = new CamCalibration();

  for (int i=0; i<multi->cams.size(); ++i) {
    calib->AddCamera(multi->cams[i]->width, multi->cams[i]->height);
  }

  geom_calib_nb_homography=0;
  glutDisplayFunc(geom_calib_draw);
  glutIdleFunc(geom_calib_idle);
}

//#define DEBUG_SHADER
/*! The paint callback during photometric calibration and augmentation. In this
 * case, we have access to 3D data. Thus, we can augment the calibration target
 * with cool stuff.
 */
static void photo_draw(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glDisable(GL_LIGHTING);
  drawBackground(frameTexture);

  if (!multi) return;

  IplImage *im = multi->model.image;
  if (!im) return;

  if (frameOK) {
    // Fetch object -> image, world->image and world -> object matrices
    CvMat *proj = multi->model.augm.GetProjectionMatrix(current_cam);
    CvMat *world = multi->model.augm.GetObjectToWorld();

    Mat3x4 moveObject, rot, obj2World, movedRT_;
    moveObject.setTranslate(im->width/2,im->height/2,-120*3/4);
    rot.setRotate(Vec3(1,0,0),2*M_PI*180.0/360.0);
    moveObject.mul(rot);
    CvMat cvMoveObject = cvMat(3,4,CV_64FC1, moveObject.m);
    CvMat movedRT=cvMat(3,4,CV_64FC1,movedRT_.m);

    double a_proj[3][4];
    for( int i = 0; i < 3; i++ )
      for( int j = 0; j < 4; j++ ) {
	a_proj[i][j] = cvmGet( proj, i, j );
	obj2World.m[i][j] = cvmGet(world, i, j);
      }
    cvReleaseMat(&proj);
    CamCalibration::Mat3x4Mul( world, &cvMoveObject, &movedRT);

    // translate into OpenGL PROJECTION and MODELVIEW matrices
    PerspectiveCamera c;
    c.loadTdir(a_proj, multi->cams[current_cam]->frame->width, multi->cams[current_cam]->frame->height);
    c.flip();
    c.setPlanes(100,1000000);
    c.setGlProjection();		
    c.setGlModelView();

    // fill the z-buffer for the calibration target
    // by drawing a transparent polygon
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glDisable(GL_CULL_FACE);
    glBegin(GL_QUADS);

#ifndef DEBUG_SHADER
    // transparent back face, for z-buffer only
    glColor4f(1,1,1,0);
    glNormal3f(0,0,-1);

    glVertex3f(multi->model.corners[0].x,multi->model.corners[0].y,0);
    glVertex3f(multi->model.corners[1].x,multi->model.corners[1].y,0);
    glVertex3f(multi->model.corners[2].x,multi->model.corners[2].y,0);
    glVertex3f(multi->model.corners[3].x,multi->model.corners[3].y,0);

#else
    // we want to relight a color present on the model image
    // with an irradiance coming from the irradiance map
    CvScalar color = cvGet2D(multi->model.image, multi->model.image->height/2, multi->model.image->width/2);
    float normal[3] = { 
      obj2World.m[0][2],
      obj2World.m[1][2],
      obj2World.m[2][2]};

    CvScalar irradiance = multi->model.map.readMap(normal);

    // the camera has some gain and bias
    const float *g = multi->model.map.getGain(current_cam);
    const float *b = multi->model.map.getBias(current_cam);

    // relight the 3 RGB channels. The bias value expects 0 black 1 white,
    // but the image are stored with a white value of 255: Conversion is required.
    for (int i=0; i<3; i++) {
      color.val[i] = (g[i]*(color.val[i]/255.0)*irradiance.val[i] + b[i]);
    }
    glColor3d(color.val[2], color.val[1], color.val[0]);

    float half[2] = {
      (multi->model.corners[0].x + multi->model.corners[1].x)/2,
      (multi->model.corners[2].x + multi->model.corners[3].x)/2,
    };
    glNormal3f(0,0,1);
    glVertex3f(multi->model.corners[0].x,multi->model.corners[0].y,0);
    glVertex3f(half[0],multi->model.corners[1].y,0);
    glVertex3f(half[1],multi->model.corners[2].y,0);
    glVertex3f(multi->model.corners[3].x,multi->model.corners[3].y,0);
#endif
    glEnd();

#ifndef DEBUG_SHADER
    // apply the object transformation matrix
    Mat3x4 w2e(c.getWorldToEyeMat());
    w2e.mul(moveObject);
    c.setWorldToEyeMat(w2e);
    c.setGlModelView();
#endif

    if (multi->model.map.isReady()) {
      glDisable(GL_LIGHTING);
#ifdef DEBUG_SHADER
      multi->model.map.enableShader(current_cam, world);
#else
      multi->model.map.enableShader(current_cam, &movedRT);
#endif
    } else {
      GLfloat light_diffuse[]  = {1.0, 1, 1, 1.0};
      GLfloat light_position[] = {500, 400.0, 500.0, 1};
      Mat3x4 w2obj;
      w2obj.setInverseByTranspose(obj2World);
      w2obj.transform(light_position, light_position);
      glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
      glLightfv(GL_LIGHT0, GL_POSITION, light_position);
      glEnable(GL_LIGHT0);
      glEnable(GL_LIGHTING);
    }

    cvReleaseMat(&world);
    {
      CvScalar c =cvGet2D(multi->model.image, 
			  multi->model.image->height/2,
			  multi->model.image->width/2);
      glColor3d(c.val[2], c.val[1], c.val[0]);
#ifndef DEBUG_SHADER
      glEnable(GL_CULL_FACE);
      if (sphereObject) {
	glCullFace(GL_BACK);
	glutSolidSphere(120, 16, 16);
      } else {
	glCullFace(GL_FRONT);
	// this rotation is the inverse of the rotation 
	// that glutSolidTeapot() does itself. It is necessary
	// to maintain modelView matrix consistant with the normal
	// transformation matrix.
	// scale and translation do not mess with normals: no need to worry.
	glRotatef(-270, 1.0, 0.0, 0.0);
	glutSolidTeapot(120);
      }
#else	
      glDisable(GL_CULL_FACE);
      glBegin(GL_QUADS);
      glNormal3f(0,0,-1);
      glVertex3f(half[0],multi->model.corners[0].y,0);
      glVertex3f(multi->model.corners[1].x,multi->model.corners[1].y,0);
      glVertex3f(multi->model.corners[2].x,multi->model.corners[2].y,0);
      glVertex3f(half[1],multi->model.corners[3].y,0);
      glEnd();
      glutSolidSphere(120, 16, 16);
#endif
    }
    if (multi->model.map.isReady())
      multi->model.map.disableShader();
    else
      glDisable(GL_LIGHTING);
  }
  if (multi->model.map.isReady()) {
    multi->model.map.map.loadTexture();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glBegin(GL_QUADS);
    {
      glColor4f(1,1,1,1);
      glTexCoord2d(0,0);
      glVertex2d(.5, 1);
      glTexCoord2d(1,0);
      glVertex2d(1,1);
      glTexCoord2d(1,1);
      glVertex2d(1,.8);
      glTexCoord2d(0,1);
      glVertex2d(.5,.8);
    }
    glEnd();
    glEnable(GL_DEPTH_TEST);
    multi->model.map.map.disableTexture();
  }


  glutSwapBuffers();
}

/*! GLUT calls this during photometric calibration or augmentation phase when
 * there's nothing else to do. This function does the 2D detection and bundle
 * adjusts the 3D pose of the calibration pattern. Then, it extracts the
 * surface normal, and pass all the lighting measurements to the LightMap
 * object. When enough is collected, the lightmap is computed.
 */
static void photo_idle()
{
  // acquire images
  multi->grabFrames();

  // detect the calibration object in every image
  // (this loop could be paralelized)
  int nbdet=0;
  for (int i=0; i<multi->cams.size(); ++i) {
    if (multi->cams[i]->detect()) nbdet++;
  }

  if(!frameTexture) frameTexture = new IplTexture;
  frameTexture->setImage(multi->cams[current_cam]->frame);

  frameOK=false;
  if (nbdet>0) {
    multi->model.augm.Clear();
    for (int i=0; i<multi->cams.size(); ++i) {
      if (multi->cams[i]->detector.object_is_detected) {
	add_detected_homography(i, multi->cams[i]->detector, multi->model.augm);
      } else {
	multi->model.augm.AddHomography();
      }
    }
    frameOK = multi->model.augm.Accomodate(4, 1e-4);
  }

  if (frameOK) {
    // fetch surface normal in world coordinates
    CvMat *mat = multi->model.augm.GetObjectToWorld();
    float normal[3];
    for (int j=0;j<3;j++) normal[j] = cvGet2D(mat, j, 2).val[0];
    cvReleaseMat(&mat);

    // During photometric calibration phase or 
    // for light update...
    if (!multi->model.map.isReady() || dynamicLight) {
      for (int i=0; i<multi->cams.size();++i) {
	// ..collect lighting measures
	if (multi->cams[i]->detector.object_is_detected) {
	  nbLightMeasures++;
	  multi->model.map.addNormal(normal, *multi->cams[i]->lc, i);
	}
      }
    }

    // when required, compute all the lighting parameters
    if (!multi->model.map.isReady() && multi->model.map.nbNormals() > 80) {
      if (multi->model.map.computeLightParams()) {
	multi->model.map.save();
      }
    }
  }
  glutPostRedisplay();
}

//! Starts photometric calibration.
static void photo_start()
{
  // allocate light collectors
  multi->allocLightCollector();

  if (cacheLight) multi->model.map.load();

  nbLightMeasures=0;
  glutIdleFunc(photo_idle);
  glutDisplayFunc(photo_draw);
}


