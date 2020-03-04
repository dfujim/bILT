TARGET     = data_generator_pp
INSTALLDIR = 
CXX        = g++
CXXFLAGS   = -march=native -O3 -Wall -fopenmp `root-config --cflags`
LDFLAGS    = 
LDLIBS     = -lyaml-cpp `root-config --glibs` -lMathMore -lMinuit2
RM         = rm -f
SOURCES    = $(shell find . -name "*.cpp")
OBJECTS    = $(patsubst %.cpp, %.o, $(SOURCES))

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJECTS) $(LDLIBS) 

depend: .depend

.depend: $(SOURCES)
	$(RM) *~ .depend
	$(CXX) $(CXXFLAGS) -MM $^>>./.depend;

install:
	mkdir -p $(INSTALLDIR)
	mv -p $(TARGET) $(INSTALLDIR)

clean:
	$(RM) $(OBJECTS)
	
cleaner: clean
	$(RM) *~ .depend

cleanest: cleaner
	$(RM) $(TARGET)

include .depend
