#include "multigrab.h"

int MultiGrab::init(bool cacheTraining)
{
  cams.clear();
  while (1) {
    CvCapture *c = cvCaptureFromCAM(cams.size());
    if (!c) break;
    cams.push_back(new Cam(c));
  }
  if (cams.size()==0) return 0;

  if (!model.buildCached(cams.size(), cams[0]->cam, cacheTraining, cams[0]->detector)) {
    cout << "model.buildCached() failed.\n";
    return 0;
  }
  for (int i=1; i<cams.size(); ++i) {
    //! TODO mem to mem copy from cams[0]
    cams[i]->detector.load("model.bmp.classifier");
  }

  return 1;
}

void MultiGrab::grabFrames()
{
  for (vector<Cam *>::iterator it=cams.begin(); it!=cams.end(); ++it)
    {
      cvGrabFrame((*it)->cam);
    }
}

void MultiGrab::allocLightCollector()
{
  for (vector<Cam *>::iterator it=cams.begin(); it!=cams.end(); ++it)
    (*it)->lc = new LightCollector(model.map.reflc);
}

void MultiGrab::Cam::setCam(CvCapture *c) {
  if (cam) cvReleaseCapture(&cam);
  if (c==0) return;
  cam = c;
  // avoid saturating the firewire bus
  cvSetCaptureProperty(cam, CV_CAP_PROP_FPS, 15);
  //cvSetCaptureProperty(cam, CV_CAP_PROP_FRAME_WIDTH, 1024);
  //cvSetCaptureProperty(cam, CV_CAP_PROP_FRAME_HEIGHT, 768);
  IplImage *f = myQueryFrame(cam);
  assert(f != 0);

  // downsample the video to accelerate computation
  width = f->width; //cvRound(cvGetCaptureProperty(cam, CV_CAP_PROP_FRAME_WIDTH));
  height = f->height; //cvRound(cvGetCaptureProperty(cam, CV_CAP_PROP_FRAME_HEIGHT));
  double fps = cvGetCaptureProperty(cam, CV_CAP_PROP_FPS);
  cout << "Camera " << width << "x" << height << " at " 
       << fps << " fps, " << f->nChannels << " channels.\n";
}

MultiGrab::Cam::~Cam() {
  if (gray && gray != frame) cvReleaseImage(&gray);
  if (cam) cvReleaseCapture(&cam);
  if (lc) delete lc;
}

// this could be run in a separate thread
bool MultiGrab::Cam::detect() 
{
  IplImage *f = myRetrieveFrame(cam);
  if (f == 0) return false;
  if (frame == 0) frame = cvCloneImage(f);
  else cvCopy(f, frame);

  // convert it to gray levels, if required
  if (frame->nChannels >1) {
    if( !gray ) 
      gray = cvCreateImage( cvGetSize(frame), IPL_DEPTH_8U, 1 );
    cvCvtColor(frame, gray, CV_RGB2GRAY);
  } else {
    gray = frame;
  }

  // run the detector
  if (detector.detect(gray)) {
    if (lc) lc->averageImage(frame, detector.H);
    return true;
  }
  return false;
}

bool add_detected_homography(int n, planar_object_recognizer &detector, CamCalibration &calib)
{
  static std::vector<CamCalibration::s_struct_points> pts;
  pts.clear();

  for (int i=0; i<detector.match_number; ++i) {
    image_object_point_match * match = detector.matches+i;
    if (match->inlier) {
      pts.push_back(CamCalibration::s_struct_points(
						    PyrImage::convCoordf(match->image_point->u, int(match->image_point->scale), 0),
						    PyrImage::convCoordf(match->image_point->v, int(match->image_point->scale), 0),
						    PyrImage::convCoordf((float)match->object_point->M[0], int(match->object_point->scale), 0),
						    PyrImage::convCoordf((float)match->object_point->M[1], int(match->object_point->scale), 0)));
    }
  }

  return calib.AddHomography(n, pts, detector.H);
}

bool add_detected_homography(int n, planar_object_recognizer &detector, CamAugmentation &a)
{
  static std::vector<CamCalibration::s_struct_points> pts;
  pts.clear();

  for (int i=0; i<detector.match_number; ++i) {
    image_object_point_match * match = detector.matches+i;
    if (match->inlier) {
      pts.push_back(CamCalibration::s_struct_points(
						    PyrImage::convCoordf(match->image_point->u, int(match->image_point->scale), 0),
						    PyrImage::convCoordf(match->image_point->v, int(match->image_point->scale), 0),
						    PyrImage::convCoordf(match->object_point->M[0], int(match->object_point->scale), 0),
						    PyrImage::convCoordf(match->object_point->M[1], int(match->object_point->scale), 0)));
    }
  }

  a.AddHomography(pts, detector.H);
  return true;
}
