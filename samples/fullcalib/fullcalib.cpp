/*! \file fullcalib.cpp
 *  \brief All in one geometric and photometric calibration example.
 *  \author Julien Pilet
 *
 *  This example interactively ask the user to show a calibration pattern,
 *  train a classifier to recognize it, calibrates the camera geometry,
 *  computes an irradiance map, and finally "augment" the video by pasting an
 *  illumination corrected polygon on the target.
 */

#include <iostream>
#include "cv.h"
#include "highgui.h"

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "calibmodel.h"

void show_result(planar_object_recognizer &recognizer, IplImage *video, IplImage *dst);
static void augment_scene(CalibModel &model, IplImage *frame, IplImage *display);
bool add_detected_homography(planar_object_recognizer &detector, CamCalibration &calib);
bool add_detected_homography(planar_object_recognizer &detector, CamAugmentation &a);
bool geometric_calibration(CalibModel &model, CvCapture *capture, bool cache);
bool photometric_calibration(CalibModel &model, CvCapture *capture, 
		int nbImages, bool cache);

void usage(const char *s) {
	cerr << "usage:\n" << s
			<< "[<cam number>|<video file>]  [-m <model image>] [-r]\n"
			"	-m	specifies model image\n"
			"	-r	do not load any data\n"
			"	-t	train a new classifier\n"
			"	-g	recompute geometric calibration\n"
			"	-l	rebuild irradiance map from scratch\n";
	exit(1);
}

int main( int argc, char** argv )
{
	CvCapture* capture = 0;

	const char *captureSrc = "0";
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
		} else {
			captureSrc = argv[i];
		}
	}

	if(strlen(captureSrc) == 1 && isdigit(captureSrc[0]))
		capture = cvCaptureFromCAM( captureSrc[0]-'0');
	else 
		capture = cvCaptureFromAVI( captureSrc ); 

	if( !capture )
	{
		cerr <<"Could not initialize capturing from " << captureSrc << " ...\n";
		return -1;
	}

	// Allocate the detector object
	CalibModel model(modelFile);

	if (!model.buildCached(capture, !redo_training)) {
		cout << "model.buildCached() failed.\n";
		return 1;
	}

	cout << "Model build. Starting geometric calibration.\n";

	if (!geometric_calibration(model,capture, !redo_geom)) {
		cerr << "Geometric calibration failed.\n";
		return 2;
	}

	cout << "Geometric calibration OK. Calibrating light...\n";
	
	photometric_calibration(model, capture, 150, !redo_lighting);
}

bool geometric_calibration(CalibModel &model, CvCapture *capture, bool cache)
{

	if (cache && model.augm.LoadOptimalStructureFromFile("camera_c.txt", "camera_r_t.txt")) {
		return true;
	}

	const char *win = "BazAR";

	IplImage*gray=0;

	cvNamedWindow(win, CV_WINDOW_AUTOSIZE);

	CamCalibration calib;

	IplImage* frame = cvQueryFrame(capture);
	calib.AddCamera(frame->width, frame->height);
	IplImage* display=cvCloneImage(frame);

	bool success=false;

	int nbHomography =0;
	while (1)
	{
		// acquire image
		frame = cvQueryFrame( capture );
		if( !frame )
			break;

		// convert it to gray levels, if required
		if (frame->nChannels >1) {
			if( !gray ) 
				gray = cvCreateImage( cvGetSize(frame), IPL_DEPTH_8U, 1 );
			cvCvtColor(frame, gray, CV_RGB2GRAY);
		} else {
			gray = frame;
		}

		// run the detector
		if (model.detector.detect(gray)) {
			add_detected_homography(model.detector, calib);
			nbHomography++;
			cout << nbHomography << " homographies.\n";
			if (nbHomography >=70) {
				if (calib.Calibrate(
							50, // max hom
							2,   // random
							3,
							3,   // padding ratio 1/2
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
		}
		show_result(model.detector, frame, display);
		cvShowImage(win, display);

		int k=cvWaitKey(10);
		if (k=='q' || k== 27)
			break;
	}

	cvReleaseImage(&display);
	if (frame->nChannels > 1)
		cvReleaseImage(&gray);
	if (success && model.augm.LoadOptimalStructureFromFile("camera_c.txt", "camera_r_t.txt")) {
		return true;
	}
	return false;
}

bool photometric_calibration(CalibModel &model, CvCapture *capture, 
		int nbImages, bool cache)
{

	if (cache) model.map.load();

	const char *win = "BazAR";

	IplImage*gray=0;

	cvNamedWindow(win, CV_WINDOW_AUTOSIZE);
	cvNamedWindow("LightMap", CV_WINDOW_AUTOSIZE);

	IplImage* frame = 0;
	IplImage* display=cvCloneImage(cvQueryFrame(capture));

	int nbHomography =0;
	LightCollector lc(model.map.reflc);
	IplImage *lightmap = cvCreateImage(cvGetSize(model.map.map.getIm()), IPL_DEPTH_8U, 
				lc.avgChannels);
	while (1)
	{
		// acquire image
		frame = cvQueryFrame( capture );
		/*
		if (frame) cvReleaseImage(&frame);
		frame = cvLoadImage("model.bmp",1);
		*/
		if( !frame )
			break;

		// convert it to gray levels, if required
		if (frame->nChannels >1) {
			if( !gray ) 
				gray = cvCreateImage( cvGetSize(frame), IPL_DEPTH_8U, 1 );
			cvCvtColor(frame, gray, CV_RGB2GRAY);
		} else {
			gray = frame;
		}

		// run the detector
		if (model.detector.detect(gray)) {
			// 2d homography found
			nbHomography++;

			// Computes 3D pose and surface normal
			model.augm.Clear();
			add_detected_homography(model.detector, model.augm);
			model.augm.Accomodate(4, 1e-4);
			CvMat *mat = model.augm.GetObjectToWorld();
			float normal[3];
			for (int j=0;j<3;j++) normal[j] = cvGet2D(mat, j, 2).val[0];
			cvReleaseMat(&mat);

			// average pixels over triangles
			lc.averageImage(frame,model.detector.H);

			// add observations
			if (!model.map.isReady())
				model.map.addNormal(normal, lc, 0);

			if (!model.map.isReady() && nbHomography >= nbImages) {
				if (model.map.computeLightParams()) {
					model.map.save();
					const float *gain = model.map.getGain(0);
					const float *bias = model.map.getBias(0);
					cout << "Gain: " << gain[0] << ", " << gain[1] << ", " << gain[2] << endl;
					cout << "Bias: " << bias[0] << ", " << bias[1] << ", " << bias[2] << endl;
				}
			} 
		} 
		
		if (model.map.isReady()) {
			double min, max;
			IplImage *map = model.map.map.getIm();
			cvSetImageCOI(map, 2);
			cvMinMaxLoc(map, &min, &max);
			cvSetImageCOI(map, 0);
			assert(map->nChannels == lightmap->nChannels);
			cvConvertScale(map, lightmap, 128, 0);
			cvShowImage("LightMap", lightmap);
			augment_scene(model, frame, display);
		} else {
			cvCopy(frame,display);
			if (model.detector.object_is_detected)
				lc.drawGrid(display, model.detector.H);
		}

		cvShowImage(win, display);

		int k=cvWaitKey(10);
		if (k=='q' || k== 27)
			break;
	}

	cvReleaseImage(&lightmap);
	cvReleaseImage(&display);
	if (frame->nChannels > 1)
		cvReleaseImage(&gray);
	return 0;
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

static void augment_scene(CalibModel &model, IplImage *frame, IplImage *display)
{
	cvCopy(frame, display);

	if (!model.detector.object_is_detected) 
		return;

	CvMat *m = model.augm.GetProjectionMatrix(0);
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

	CvMat *o2w = model.augm.GetObjectToWorld();
	float normal[3];
	for (int j=0;j<3;j++)
		normal[j] = cvGet2D(o2w, j, 2).val[0];
	cvReleaseMat(&o2w);

	// we want to relight a color present on the model image
	// with an irradiance coming from the irradiance map
	CvScalar color = cvGet2D(model.image, model.image->height/2, model.image->width/2);
	CvScalar irradiance = model.map.readMap(normal);

	// the camera has some gain and bias
	const float *g = model.map.getGain(0);
	const float *b = model.map.getBias(0);

	// relight the 3 RGB channels. The bias value expects 0 black 1 white,
	// but the image are stored with a white value of 255: Conversion is required.
	for (int i=0; i<3; i++) {
		color.val[i] = 255.0*(g[i]*(color.val[i]/255.0)*irradiance.val[i] + b[i]);
	}

	// draw a filled polygon with the relighted color
	cvFillConvexPoly(display, projPts, 4, color);
}

bool add_detected_homography(planar_object_recognizer &detector, CamCalibration &calib)
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

	return calib.AddHomography(0, pts, detector.H);
}

bool add_detected_homography(planar_object_recognizer &detector, CamAugmentation &a)
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
