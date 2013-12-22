
if FULLGL
openglprogs=multigl
else
openglprogs=
endif

noinst_PROGRAMS=multicam $(openglprogs)

multicam_CXXFLAGS=@OPENCV_CFLAGS@ -I@top_srcdir@/garfeild -I@top_srcdir@/starter
multicam_LDADD= ../../garfeild/libgarfeild.a ../../starter/libstarter.a \
		 @OPENCV_LIBS@ @GLEW_LIBS@ @GLU_LIBS@
multicam_SOURCES=multicam.cpp calibmodel.cpp calibmodel.h \
		 multigrab.h multigrab.cpp

multigl_CXXFLAGS=@OPENCV_CFLAGS@ -I@top_srcdir@/garfeild -I@top_srcdir@/starter
multigl_LDADD= ../../garfeild/libgarfeild.a ../../starter/libstarter.a \
		@OPENCV_LIBS@ @GLEW_LIBS@ @GLUT_LIBS@ @GLU_LIBS@
multigl_SOURCES=multigl.cpp calibmodel.cpp calibmodel.h \
		 multigrab.h multigrab.cpp
