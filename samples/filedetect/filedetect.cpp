/*
Copyright 2005, 2006 Computer Vision Lab, 
Ecole Polytechnique Federale de Lausanne (EPFL), Switzerland. 
All rights reserved.

This file is part of BazAR.

BazAR is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

BazAR is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
BazAR; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/
/*! \file filedetect.cpp
* \brief A planar_object_recognizer example.
*
* This very simple example loads a planar_object_recognizer object and detects
* a pattern in the images passed on the command line.
*/
#include <iostream>

// include everything garfeild needs
#include <garfeild.h>

// image loading and saving with OpenCV
#include <highgui.h>

using namespace std;

int main(int argc, char *argv[])
{
	// model.png is the default model image file name.
	char *modelFile= "model.jpg";

	if (argc<2) {
		cerr << "Usage: " << argv[0] 
			<< " [-m <model image>] <image> [<image> ...]\n";
		return -1;
	}

	// search for model name
	for (int i=1; i<argc-1; i++) {
		if (strcmp(argv[i], "-m") ==0) {
			modelFile = argv[i+1];
			break;
		}
	}

	// Allocate the detector object
	planar_object_recognizer detector;

	// A lower threshold will allow detection in harder conditions, but
	// might lead to false positives.
	detector.match_score_threshold=.03f;

	detector.ransac_dist_threshold = 5;
	detector.max_ransac_iterations = 500;
	detector.non_linear_refine_threshold = 1.5;
	detector.min_view_rate=.2;
	detector.views_number = 100;

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
		cerr << modelFile << ": Error while loading model image or classifier!\n";
		return -2;
	}

	// The detector is now ready. Load input images.
	for (int i=1; i<argc; ++i) {
		if (argv[i][0] == '-') { ++i; continue; }
		IplImage *im = cvLoadImage(argv[i],0);
		if (!im) {
			cerr << argv[i] << ": unable to load image.\n";
			continue;
		} 
		if (detector.detect(im)) {
			cout << argv[i] << ": detection succeeded!\n";

			// save an image showing the result
			IplImage *result = detector.create_result_image(im, false,true,false,true);
			char fn[512]; sprintf(fn,"result_%s", argv[i]);
			cvSaveImage(fn, result);
			cvReleaseImage(&result);
		} else {
			cout << argv[i] << ": detection failed.\n";
		}
	}
	return 0;
}
