

noinst_PROGRAMS=fullcalib

fullcalib_CXXFLAGS=@OPENCV_CFLAGS@ -I@top_srcdir@/garfeild -I@top_srcdir@/starter
fullcalib_LDADD=../../garfeild/libgarfeild.a ../../starter/libstarter.a \
		 @OPENCV_LIBS@ @GLEW_LIBS@ @GLU_LIBS@
fullcalib_SOURCES=fullcalib.cpp calibmodel.cpp calibmodel.h

