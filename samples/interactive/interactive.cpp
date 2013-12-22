/*!\file
 * \brief An interactive example of 2D planar object training and detection.
 * This example uses HighGUI to interactively propose to the user to create a
 * planar model. It then shows live 2D detection of this object.
 */
#include <iostream>
#include "cv.h"
#include "highgui.h"
#include <garfeild.h>

#ifdef WIN32
#include <sys/timeb.h>
#else
#include <config.h>
#include <sys/time.h>
#endif

char *modelFile="model.jpg";

void acquire_model(CvCapture *capture, planar_object_recognizer &detector); 
void show_result(planar_object_recognizer &recognizer, IplImage *video, IplImage **dst);

using namespace std;

void usage(const char *s) {
	cerr << "usage:\n" << s
			<< "[<cam number>|<video file>]  [-m <model image>]\n";
	exit(1);
}

int main( int argc, char** argv )
{
	CvCapture* capture = 0;

	//cvSetErrMode(CV_ErrModeParent);

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

	cout << "Instructions:\n" 
		" 1 - present a textured planar object to the camera. \n"
		" 2 - push space to freeze the image\n"
		" 3 - if the frozen image makes a good model image, press 'y'.\n"
		"     Otherwise, press space to restart the video and take another shot. \n";

	// Allocate the detector object
	planar_object_recognizer detector;


	// Train or load classifier
	if(!detector.build_with_cache(
				string(modelFile), // mode image file name
				400,               // maximum number of keypoints on the model
				32,                // patch size in pixels
				3,                 // yape radius. Use 3,5 or 7.
				16,                // number of trees for the classifier. Somewhere between 12-50
				3                  // number of levels in the gaussian pyramid
				))
	{
		// interactively acquire a model image
		acquire_model(capture, detector);
	}

	// A lower threshold will allow detection in harder conditions, but
	// might lead to false positives.
	detector.match_score_threshold=.03f;

	const char *win = "Bazar";

	IplImage* display=0;
	IplImage*gray=0;

	cvNamedWindow(win, CV_WINDOW_AUTOSIZE);

#ifdef WIN32
  struct timeb last, now;
  ftime(&last);
#else
  struct timeval last, now;
  gettimeofday(&last,0);
#endif

  int nbFrames=0;
	
	for(;;)
	{
		IplImage* frame = 0;

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
		}
		show_result(detector, frame, &display);
		cvShowImage(win, display);
		//cvShowImage(win, frame);

		if (nbFrames==4) {
			nbFrames=0;

      double duration;

#ifdef WIN32
      ftime(&now);
      duration = ( double(( now.time-last.time ) * 1e4) 
                +  double( now.millitm-last.millitm ) ) / 1e4;
#else
      gettimeofday(&now,0);
      duration = double(now.tv_sec-last.tv_sec)
               + double(now.tv_usec-last.tv_usec)/1e6;
#endif

			last = now;

			cout << "FPS: " << 5.0/duration << endl;
		} else {
			nbFrames++;
		}

		if( cvWaitKey(1) >= 0 )
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

void acquire_model(CvCapture *capture, planar_object_recognizer &detector)
{

	const char *win = "Bazar";

	cvNamedWindow(win, CV_WINDOW_AUTOSIZE);

	bool pause=false;
	IplImage *frame;
	IplImage *shot=0;

	bool accepted =false;
	while (!accepted) {

		if (!pause) {
			frame = cvQueryFrame(capture);
			cvShowImage(win, frame);
		}

		int k = cvWaitKey(pause ? 50 : 1);
		switch (k) {
			case ' ': 
				  pause = !pause; 
				  if (pause) {
					  if (shot) cvCopy(frame,shot);
					  else shot = cvCloneImage(frame);
					  cvShowImage(win, shot);
				  }
				  break;
			case 'y':
			case '\n': if (pause && shot) accepted=true; break;
			case 'q': exit(0); break;
		}
	}

	cvSaveImage(modelFile, shot);
	detector.build(shot, 400, 32, 3, 16, 3);
	detector.save(string(modelFile)+".classifier");
	cvReleaseImage(&shot);
}
