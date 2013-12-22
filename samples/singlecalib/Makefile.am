

noinst_PROGRAMS=singlecalib augment3d

singlecalib_CXXFLAGS=-I@top_srcdir@/garfeild -I@top_srcdir@/starter @OPENCV_CFLAGS@ 
singlecalib_LDADD=../../garfeild/libgarfeild.a ../../starter/libstarter.a @OPENCV_LIBS@
singlecalib_SOURCES=singlecalib.cpp

augment3d_CXXFLAGS=-I@top_srcdir@/garfeild -I@top_srcdir@/starter @OPENCV_CFLAGS@ 
augment3d_LDADD=../../garfeild/libgarfeild.a ../../starter/libstarter.a @OPENCV_LIBS@ 
augment3d_SOURCES=augment3d.cpp

