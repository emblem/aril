#ifndef _MULTIGRAB_H
#define _MULTIGRAB_H

#include "calibmodel.h"

class MultiGrab {
public:

	CalibModel model;

	MultiGrab(const char *modelfile="model.bmp") : model(modelfile) {}

	int init(bool cacheTraining);
	void grabFrames();
	void allocLightCollector();

	struct Cam {
		CvCapture *cam;
		IplImage *frame, *gray;
		int width,height;
		planar_object_recognizer detector;
		LightCollector *lc;

		void setCam(CvCapture *c);
		bool detect();

		Cam(CvCapture *c=0, IplImage *f=0) 
		{
			width=0;
			height=0;
			cam=0;
			lc=0;
			if (c) setCam(c);
			frame=f;
			gray=0;
		}
		~Cam();
	};

	std::vector<Cam *> cams;
};

bool add_detected_homography(int n, planar_object_recognizer &detector, CamCalibration &calib);
bool add_detected_homography(int n, planar_object_recognizer &detector, CamAugmentation &a);
IplImage *myQueryFrame(CvCapture *capture);
IplImage *myRetrieveFrame(CvCapture *capture);

#endif
