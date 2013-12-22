#ifndef CALIBMODEL_H
#define CALIBMODEL_H

#include <cv.h>
#include <highgui.h>
#include <garfeild.h>

class CalibModel {
 public:
  IplImage *image;
  CvPoint corners[4];

  planar_object_recognizer detector;
  LightMap map;
  CamAugmentation augm;

  CalibModel(const char *modelfile = "model.bmp");
  ~CalibModel();

  bool buildCached(CvCapture *capture, bool cache);

 private:
  const char *win;
  const char *modelfile;

  enum State { TAKE_SHOT, CORNERS };
  State state;
  int grab;
  static void onMouseStatic(int event, int x, int y, int flags, void* param);
  void onMouse(int event, int x, int y, int flags);
  bool interactiveSetup(CvCapture *capture);
};


#endif
