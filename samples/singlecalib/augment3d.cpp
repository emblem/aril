/*! \file augment3d.cpp
 *  \brief A live 3D augmentation example
 *  \author Julien Pilet
 *
 * This example shows how to draw virtual 3D axis on a real object. Camera
 * geometric calibration is loaded from disk. The file singlecalib.cpp shows
 * how to compute such a calibration.
 */
#include <iostream>
#include "cv.h"
#include "highgui.h"
#include <garfeild.h>

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

char *modelFile="model.jpg";

bool add_detected_homography(planar_object_recognizer &detector, CamAugmentation &calib);
void show_result(CamAugmentation &augment, IplImage **dst);

void usage(const char *s) {
	cerr << "usage:\n" << s
			<< "[<cam number>|<video file>]  [-m <model image>]\n";
	exit(1);
}

int main( int argc, char** argv )
{
	CvCapture* capture = 0;

	const char *captureSrc = "0";

	// parse command line
	for (int i=1; i<argc; i++) {
		if (strcmp(argv[i], "-m") ==0) {
			if (i==argc-1) usage(argv[0]);
			modelFile = argv[i+1];
			i++;
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

	// fine tuning for accuracy
	detector.ransac_dist_threshold = 5;
	detector.max_ransac_iterations = 800;
	detector.non_linear_refine_threshold = 5;

	// Train or load classifier
	if(!detector.build_with_cache(
				string(modelFile), // mode image file name
				400,               // maximum number of keypoints on the model
				32,                // patch size in pixels
				7,                 // yape radius. Use 3,5 or 7.
				32,                // number of trees for the classifier. Somewhere between 12-50
				4                  // number of levels in the gaussian pyramid
				))
	{
		cerr << "Unable to load the model image "
		       << modelFile <<"	or its classifier.\n";
		return -1;
	}

	// A lower threshold will allow detection in harder conditions, but
	// might lead to false positives.
	detector.match_score_threshold=.1f;

	const char *win = "Bazar";

	IplImage* display=0;
	IplImage*gray=0;

	cvNamedWindow(win, 0);

	CamAugmentation augment;

	if (!augment.LoadOptimalStructureFromFile("camera_c.txt", "camera_r_t.txt")) {
		cerr << "Unable to load calibration data\n";
		return -2;
	}

	for(;;)
	{
		// acquire image
		IplImage *frame = cvQueryFrame( capture );
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

		if (display==0) display=cvCloneImage(frame);
		else cvCopy(frame, display);


		// run the detector
		if (detector.detect(gray)) {
			// start on a new frame
			augment.Clear();

			// we have only 1 camera
			add_detected_homography(detector, augment);

			// bundle adjust
			augment.Accomodate(4, 1e-4);


		}
		show_result(augment, &display);
		/*
		int patch_size = detector.forest->image_width;

		for(int i = 0; i < detector.detected_point_number; i++) {
		  cvCircle(display, 
			   cvPoint((int)PyrImage::convCoordf(detector.detected_points[i].u, int(detector.detected_points[i].scale), 0), 
				   (int)PyrImage::convCoordf(detector.detected_points[i].v, int(detector.detected_points[i].scale), 0)),
			   (int)PyrImage::convCoordf(patch_size/2.f, int(detector.detected_points[i].scale), 0), 
			   mcvRainbowColor(int(detector.detected_points[i].scale)), 1);
			   }*/


		
		int patch_size = detector.forest->image_width;
		float min = 10000;
		float max = 0;
		for(int i = 0; i < detector.match_number; i++) {
		  float score = detector.matches[i].score;
		  min = score < min ? score : min;
		  max = score > max ? score : max;

		  if(detector.matches[i].score < detector.match_score_threshold ) continue;
		  cvCircle(display, 
			   cvPoint((int)PyrImage::convCoordf(detector.matches[i].image_point->u, int(detector.matches[i].image_point->scale), 0), 
				   (int)PyrImage::convCoordf(detector.matches[i].image_point->v, int(detector.matches[i].image_point->scale), 0)),
			   (int)PyrImage::convCoordf(patch_size/2.f, int(detector.matches[i].image_point->scale), 0), 
			   mcvRainbowColor(int(detector.matches[i].image_point->scale)), 1);
		}
		cout << "Min: " << min << "\t Max:" << max << endl;
		
		cvShowImage(win,display);
		//cvShowImage(win, frame);

		if( cvWaitKey(10) >= 0 )
			break;
	}

	cvReleaseCapture( &capture );
	cvDestroyWindow(win);

	return 0;
}

void show_result(CamAugmentation &augment, IplImage **dst)
{

	CvMat *m = augment.GetProjectionMatrix(0);
	if (!m) return;

	double w =(*dst)->width/2.0;
	double h =(*dst)->height/2.0;

	// 3D coordinates of an object
	double pts[4][4] = {
		{w,h,0, 1},
		{2*w,h,0, 1},
		{w,2*h,0, 1},
		{w,h,-w-h, 1}
	};
	double projected[3][4];
	CvMat ptsMat, projectedMat;
	cvInitMatHeader(&ptsMat, 4, 4, CV_64FC1, pts);
	cvInitMatHeader(&projectedMat, 3, 4, CV_64FC1, projected);

	// project the 4 3D points
	cvGEMM(m, &ptsMat, 1, 0, 0, &projectedMat, CV_GEMM_B_T );
	for (int i=0; i<4; i++) {
		projected[0][i] /= projected[2][i];
		projected[1][i] /= projected[2][i];
	}

	// draw the projected lines
	cvLine(*dst, cvPoint((int)projected[0][0], (int)projected[1][0]),
			cvPoint((int)projected[0][1], (int)projected[1][1]), CV_RGB(0,255,0));
	cvLine(*dst, cvPoint((int)projected[0][0], (int)projected[1][0]),
			cvPoint((int)projected[0][2], (int)projected[1][2]), CV_RGB(255,0,0));
	cvLine(*dst, cvPoint((int)projected[0][0], (int)projected[1][0]),
			cvPoint((int)projected[0][3], (int)projected[1][3]), CV_RGB(0,0,255));

	
	cvReleaseMat(&m);
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
