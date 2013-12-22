/*! \file singlecalib.cpp
 * \brief A single camera geometric calibration interactive example.
 * \author Julien Pilet
 * In this example, the user is asked to prepare a calibration pattern. After
 * moving it around, geometric calibration of the camera is computed and saved
 * on disk.
 * The file augment3d.cpp shows how to use the resulting calibration to put a
 * virtual 3D object on the target.
 */
#include <iostream>
#include "cv.h"
#include "highgui.h"
#include <garfeild.h>

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

char *modelFile="model.jpg";

IplImage *acquire_model(CvCapture *capture);
void show_result(planar_object_recognizer &recognizer, IplImage *video, IplImage **dst);
bool add_detected_homography(planar_object_recognizer &detector, CamCalibration &calib);

void usage(const char *s) {
	cerr << "usage:\n" << s
			<< "[<cam number>|<video file>]  [-m <model image>] [-r]\n"
			"	-m	specifies model image\n"
			"	-r	do not load cached model image\n";
	exit(1);
}

int main( int argc, char** argv )
{
	CvCapture* capture = 0;

	const char *captureSrc = "0";
	bool relearn=false;

	// parse command line
	for (int i=1; i<argc; i++) {
		if (strcmp(argv[i], "-m") ==0) {
			if (i==argc-1) usage(argv[0]);
			modelFile = argv[i+1];
			i++;
		} else if (strcmp(argv[i], "-r")==0) {
			relearn=true;
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
	planar_object_recognizer detector;

	detector.ransac_dist_threshold = 5;
	detector.max_ransac_iterations = 800;
	detector.non_linear_refine_threshold = 1.5;

	// Train or load classifier
	if(relearn || !detector.build_with_cache(
				string(modelFile), // mode image file name
				400,               // maximum number of keypoints on the model
				32,                // patch size in pixels
				7,                 // yape radius. Use 3,5 or 7.
				32,                // number of trees for the classifier. Somewhere between 12-50
				4                  // number of levels in the gaussian pyramid
				))
	{
		// interactively acquire a model image
		IplImage *shot = acquire_model(capture);
		cvSaveImage(modelFile, shot);
		detector.build(shot, 400, 32, 7, 32, 4);
		detector.save(string(modelFile)+".classifier");
		cvReleaseImage(&shot);
	}

	// A lower threshold will allow detection in harder conditions, but
	// might lead to false positives.
	detector.match_score_threshold=.03f;

	const char *win = "Bazar";

	IplImage* display=0;
	IplImage*gray=0;

	cvNamedWindow(win, 0);

	CamCalibration calib;

	IplImage* frame = cvQueryFrame(capture);
	calib.AddCamera(frame->width, frame->height);

	int nbHomography =0;
	for(;;)
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
		if (detector.detect(gray)) {
			add_detected_homography(detector, calib);
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
						   )) {
					calib.PrintOptimizedResultsToFile1();
					break;
				}
			}
		}
		show_result(detector, frame, &display);
		cvShowImage(win, display);
		//cvShowImage(win, frame);

		if( cvWaitKey(10) >= 0 )
			break;
	}

	cvReleaseCapture( &capture );
	cvDestroyWindow(win);

	return 0;
}

void show_result(planar_object_recognizer &detector, IplImage *video, IplImage **dst)
{
	if (*dst==0) *dst=cvCloneImage(video);
	else cvCopy(video, *dst);

	if (detector.object_is_detected) {
		for (int i=0; i<detector.match_number; ++i) {

			image_object_point_match * match = detector.matches+i;
			if (match->inlier) {
			cvCircle(*dst,
				cvPoint((int) (PyrImage::convCoordf(match->image_point->u, 
							int(match->image_point->scale), 0)),
					(int)(PyrImage::convCoordf(match->image_point->v, 
							int(match->image_point->scale), 0))),
				3, CV_RGB(0,255,0), -1, 8,0);
			}
	}
	}
}

static void putText(IplImage *im, const char *text, CvPoint p, CvFont *f1, CvFont *f2)
{
	cvPutText(im,text,p,f2, cvScalarAll(0));
	cvPutText(im,text,p,f1, cvScalarAll(255));
}

IplImage *acquire_model(CvCapture *capture)
{

	const char *win = "Bazar";

	CvFont font, fontbold;

	cvInitFont( &font, CV_FONT_HERSHEY_PLAIN, 1, 1);
	cvInitFont( &fontbold, CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 5);

	cvNamedWindow(win, 0);

	bool pause=false;
	IplImage *frame;
	IplImage *shot=0, *text=0;

	bool accepted =false;
	while (!accepted) {

		if (!pause) {
			frame = cvQueryFrame(capture);
			if (!text) text=cvCloneImage(frame);
			else cvCopy(frame,text);
			putText(text,"Please take a frontal view of a", cvPoint(3,20), &font, &fontbold);
			putText(text,"textured planar surface and press space", cvPoint(3,40), &font, &fontbold);
			cvShowImage(win, text);
		}

		char k = cvWaitKey(pause ? 0 : 10);
		switch (k) {
			case 'n': pause=false; break;
			case ' ': 
				  pause = !pause; 
				  if (pause) {
					  if (shot) cvCopy(frame,shot);
					  else shot = cvCloneImage(frame);
					  cvCopy(shot,text);
					  putText(text,"Image OK? (y/n)", cvPoint(3,20), &font, &fontbold);
					  cvShowImage(win, text);
				  }
				  break;
			case 'y':
			case '\n': if (pause && shot) accepted=true; break;
			case 'q': exit(0); break;
			case -1: break;
			default: cerr << k << ": what?\n";
		}
	}

	cvReleaseImage(&text);
	return shot;
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
