

EXTRA_DIST=a.jpg b.jpg c.jpg d.jpg e.jpg f.jpg model.jpg printme.jpg

noinst_PROGRAMS=filedetect

filedetect_CXXFLAGS= -I@top_srcdir@/garfeild -I@top_srcdir@/starter @OPENCV_CFLAGS@
filedetect_LDADD= ../../garfeild/libgarfeild.a ../../starter/libstarter.a @OPENCV_LIBS@ 

filedetect_SOURCES=filedetect.cpp
