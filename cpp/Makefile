# Note that we're abusing the fact that make does not complain about directories that are not actually here.
# Therefore, we are lazy and combine paths for both mac local and cms cluster

CC = g++
ROOT_FOLDER = /home/stzheng/projects/robust_multitask/cpp
INC_DIR = \
-I/home/ubuntu/robustmultitask_devstack/lib \
-I/home/ubuntu/robustmultitask_devstack/lib/boost_1_57_0 \
-I/home/stzheng/lib/rapidjson/include \
-I/home/stzheng/lib/boost_1_57_0 \
-I/home/stzheng/projects \
-I$(ROOT_FOLDER)

LIB_DIR = \
-L/home/ubuntu/robustmultitask_devstack/lib/boost_1_57_0/stage/lib \
-L/home/stzheng/lib/boost_1_57_0/stage/lib

LDFLAGS = -lboost_thread -lboost_filesystem -lboost_system
CFLAGS = -w -g $(LIB_DIR) $(INC_DIR) -std=c++11 -m64
OPTFLAGS = -Ofast
SOURCES = \
$(wildcard config/*.cc) \
$(wildcard core/*.cc) \
$(wildcard util/*.cc) \
$(wildcard test/*.cc)

OBJECTS_DIR = obj
_OBJECTS = $(SOURCES:.cc=.o)
OBJECTS = $(patsubst %,$(OBJECTS_DIR)/%,$(_OBJECTS))

EXECUTABLE = train_robust_multitask

$(OBJECTS_DIR)/%.o : %.cc
	mkdir -p $(OBJECTS_DIR)
	mkdir -p $(@D)
	$(CC) $(OPTFLAGS) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OPTFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)

all: $(EXECUTABLE)

clean:
	$(RM) -rf $(OBJECTS_DIR) *~ $(EXECUTABLE)

depend: $(SOURCES)
	makedepend $(INC_DIR) $^

# DO NOT DELETE THIS LINE -- make depend needs it

debug: $(OBJECTS)
	$(CC) $(CFLAGS) -gdwarf2 -DDEBUG $^ -o $(EXECUTABLE) $(LDFLAGS)