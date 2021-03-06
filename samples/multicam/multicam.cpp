
#include <iostream>
#include <vector>
#include <cv.h>
#include <highgui.h>

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "multigrab.h"

void show_result(planar_object_recognizer &recognizer, IplImage *video, IplImage *dst);
static void augment_scene(int cam, CalibModel &model, IplImage *frame, IplImage *display);
bool geometric_calibration(MultiGrab &multi, bool cache);
bool photometric_calibration(MultiGrab &multi, int nbImages, bool cache);


void usage(const char *s) {
  cerr << "usage:\n" << s
       << "[-m <model image>] [-r]\n"
    "	-m	specifies model image\n"
    "	-r	do not load any data\n"
    "	-t	train a new classifier\n"
    "	-g	recompute geometric calibration\n"
    "	-l	rebuild irradiance map from scratch\n";
  exit(1);
}

int main( int argc, char** argv )
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

  MultiGrab multi;

  if( multi.init(!redo_training) ==0 )
    {
      cerr <<"Initialization error.\n";
      return -1;
    }

  cout << "Starting geometric calibration.\n";

  if (!geometric_calibration(multi, !redo_geom)) {
    cerr << "Geometric calibration failed.\n";
    return 2;
  }

  cout << "Geometric calibration OK. Calibrating light...\n";
	
  // start collecting light measurements
  multi.allocLightCollector();

  photometric_calibration(multi, 150, !redo_lighting);
}

bool geometric_calibration(MultiGrab &multi, bool cache)
{

  if (cache && multi.model.augm.LoadOptimalStructureFromFile("camera_c.txt", "camera_r_t.txt")) {
    return true;
  }

  const char *win = "BazAR";

  cvNamedWindow(win, CV_WINDOW_AUTOSIZE);

  // construct a CamCalibration object and register all the cameras
  CamCalibration calib;

  for (int i=0; i<multi.cams.size(); ++i) {
    calib.AddCamera(multi.cams[i]->width, multi.cams[i]->height);
  }

  IplImage *display=0;
  bool success=false;
  bool end=false;

  int dispCam=0;
  int nbHomography =0;
  while (!end)
    {
      // acquire images
      multi.grabFrames();

      // detect the calibration object in every image
      // (this loop could be paralelized)
      int nbdet=0;
      for (int i=0; i<multi.cams.size(); ++i) {
	if (multi.cams[i]->detect()) nbdet++;
      }
		
      if (nbdet>0) {
	for (int i=0; i<multi.cams.size(); ++i) {
	  if (multi.cams[i]->detector.object_is_detected) {
	    add_detected_homography(i, multi.cams[i]->detector, calib);
	  } else {
	    calib.AddHomography(i);
	  }
	}
	nbHomography++;
      }

      if (nbHomography >=200) {
	if (calib.Calibrate(
			    120, // max hom
			    (multi.cams.size() > 1 ? 1:2),   // padding or random
			    3,
			    .5,   // padding ratio 1/2
			    0,
			    0,
			    0.0078125,	//alpha
			    0.9,		//beta
			    0.001953125,//gamma
			    12,	  // iter
			    0.05, //eps
			    3   //postfilter eps
			    )) 
	  {
	    calib.PrintOptimizedResultsToFile1();
	    success=true;
	    break;
	  }
      }
      if (display==0) display = cvCreateImage(cvGetSize(multi.cams[dispCam]->frame), IPL_DEPTH_8U, 3);
      show_result(multi.cams[dispCam]->detector, multi.cams[dispCam]->frame, display);
      cvShowImage(win, display);

      int k=cvWaitKey(10);
      switch (k) {
      case 'q':
      case 27: end=true; break;
      case 'n': if(dispCam < multi.cams.size()-1) {
	  cvReleaseImage(&display);
	  ++dispCam; 
	}
	cout << "Current cam: " << dispCam << endl;
	break;
      case 'p': if(dispCam > 0) {
	  cvReleaseImage(&display);
	  --dispCam;
	}
	cout << "Current cam: " << dispCam << endl;
	break;
      case -1: break;
      default: cout << (char)k <<": What ?\n";
      }
    }

  if (display) cvReleaseImage(&display);
  if (success && multi.model.augm.LoadOptimalStructureFromFile("camera_c.txt", "camera_r_t.txt")) {
    return true;
  }
  return false;
}

bool photometric_calibration(MultiGrab &multi, int nbImages, bool cache)
{
  CalibModel &model(multi.model);

  if (cache) model.map.load();

  const char *win = "BazAR";

  cvNamedWindow(win, CV_WINDOW_AUTOSIZE);

  IplImage *display=0;
  bool success=false;
  bool end=false;

  int dispCam=0;
  int nbLightMeasures =0;
  while (!end)
    {
      // acquire images
      multi.grabFrames();

      // detect the calibration object in every image
      // (this loop could be paralelized)
      int nbdet=0;
      for (int i=0; i<multi.cams.size(); ++i) {
	if (multi.cams[i]->detect()) nbdet++;
      }
		
      bool frameOK=false;
      if (nbdet>0) {
	model.augm.Clear();
	for (int i=0; i<multi.cams.size(); ++i) {
	  if (multi.cams[i]->detector.object_is_detected) {
	    add_detected_homography(i, multi.cams[i]->detector, model.augm);
	  } else {
	    model.augm.AddHomography();
	  }
	}
	frameOK = model.augm.Accomodate(4, 1e-4);
      }

      if (display==0) display = cvCreateImage(cvGetSize(multi.cams[dispCam]->frame), IPL_DEPTH_8U, 3);

      if (frameOK) {
	// fetch surface normal in world coordinates
	CvMat *mat = model.augm.GetObjectToWorld();
	float normal[3];
	for (int j=0;j<3;j++) normal[j] = cvGet2D(mat, j, 2).val[0];
	cvReleaseMat(&mat);

	// collect lighting measures
	for (int i=0; i<multi.cams.size();++i) {
	  if (multi.cams[i]->detector.object_is_detected) {
	    nbLightMeasures++;
	    model.map.addNormal(normal, *multi.cams[i]->lc, i);
	  }
	}
	if (!model.map.isReady() && nbLightMeasures > 40) {
	  if (model.map.computeLightParams()) {
	    model.map.save();
	  }
	}

	augment_scene(dispCam, model, multi.cams[dispCam]->frame, display);
      } else {
	cvCopy( multi.cams[dispCam]->frame, display);
      }
      cvShowImage(win, display);

      int k=cvWaitKey(10);
      switch (k) {
      case 'q':
      case 27: end=true; break;
      case 'n': if(dispCam < multi.cams.size()-1) {
	  cvReleaseImage(&display);
	  ++dispCam; 
	}
	cout << "Current cam: " << dispCam << endl;
	break;
      case 'p': if(dispCam > 0) {
	  cvReleaseImage(&display);
	  --dispCam;
	}
	cout << "Current cam: " << dispCam << endl;
	break;
      case -1: break;
      default: cout << (char)k <<": What ?\n";
      }
    }

  if (display) cvReleaseImage(&display);
  if (success && model.augm.LoadOptimalStructureFromFile("camera_c.txt", "camera_r_t.txt")) {
    return true;
  }
  return false;


  return false;
}

void show_result(planar_object_recognizer &detector, IplImage *video, IplImage *dst)
{
  cvCopy(video, dst);

  if (detector.object_is_detected) {
    for (int i=0; i<detector.match_number; ++i) {

      image_object_point_match * match = detector.matches+i;
      if (match->inlier) {
	cvCircle(dst,
		 cvPoint((int) (PyrImage::convCoordf(match->image_point->u, 
						     int(match->image_point->scale), 0)),
			 (int)(PyrImage::convCoordf(match->image_point->v, 
						    int(match->image_point->scale), 0))),
		 3, CV_RGB(0,255,0), -1, 8,0);
      }
    }
  }
}

static void augment_scene(int cam, CalibModel &model, IplImage *frame, IplImage *display)
{
  cvCopy(frame, display);

  CvMat *m = model.augm.GetProjectionMatrix(cam);
  if (!m) return;

  double pts[4][4];
  double proj[4][4];
  CvMat ptsMat, projMat;
  cvInitMatHeader(&ptsMat, 4, 4, CV_64FC1, pts);
  cvInitMatHeader(&projMat, 3, 4, CV_64FC1, proj);
  for (int i=0; i<4; i++) {
    pts[0][i] = model.corners[i].x;
    pts[1][i] = model.corners[i].y;
    pts[2][i] = 0;
    pts[3][i] = 1;
  }
  cvMatMul(m, &ptsMat, &projMat);
  cvReleaseMat(&m);

  CvPoint projPts[4];
  for (int i=0;i<4; i++) {
    projPts[i].x = cvRound(proj[0][i]/proj[2][i]);
    projPts[i].y = cvRound(proj[1][i]/proj[2][i]);
  }

  CvScalar color = cvScalar(128,128,128,128);

  if (model.map.isReady()) {
    CvMat *o2w = model.augm.GetObjectToWorld();
    float normal[3];
    for (int j=0;j<3;j++)
      normal[j] = cvGet2D(o2w, j, 2).val[0];
    cvReleaseMat(&o2w);

    // we want to relight a color present on the model image
    // with an irradiance coming from the irradiance map
    color = cvGet2D(model.image, model.image->height/2, model.image->width/2);
    CvScalar irradiance = model.map.readMap(normal);

    // the camera has some gain and bias
    const float *g = model.map.getGain(cam);
    const float *b = model.map.getBias(cam);

    // relight the 3 RGB channels. The bias value expects 0 black 1 white,
    // but the image are stored with a white value of 255: Conversion is required.
    for (int i=0; i<3; i++) {
      color.val[i] = 255.0*(g[i]*(color.val[i]/255.0)*irradiance.val[i] + b[i]);
    }
  } 

  // draw a filled polygon with the relighted color
  cvFillConvexPoly(display, projPts, 4, color);
}

