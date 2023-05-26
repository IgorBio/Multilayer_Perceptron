APP=MLP

.PHONY: all app tests clean cppcheck style

CC=g++ -std=c++17 -Wall -Werror -Wextra
CHECKFLAGS=-lgtest
OPENMP=-fopenmp
CURRENTDIR = $(shell pwd)
BUILD_DIR=build
APP_DIR=MLP
REPORTDIR=gcov_report
GCOV=--coverage
OPEN=
FILTER=
CPPCHECKFLAGS = --enable=all --suppress=missingIncludeSystem --language=c++ --std=c++17
TEST_LIB:=./tests/tests_main.cc \
	./model/utility/matrix_operations.cc

OS = $(shell uname)

ifeq ($(OS), Linux)
	CC+=-D OS_LINUX -g -s
	CHECKFLAGS+=-lpthread
	CHECK_LEAKS=CK_FORK=no valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --log-file=log.txt
	OPEN=xdg-open
	DIR=
	QTFLAGS=-spec linux-g++
else
	CC+=-D OS_MAC
	CHECK_LEAKS=CK_FORK=no leaks --atExit --
	FILTER=--gtest_filter=-*.Exception*
	OPEN=open
	DIR=/$(APP).app
	QTFLAGS=
endif

all: build

build: 
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR); cmake ..; make

rebuild: clean build

install: build uninstall
	@mkdir -p $(APP_DIR)
	@cp -r $(BUILD_DIR)$(DIR) $(APP_DIR)

dist: install
	tar -czvf $(APP_DIR).tgz $(APP_DIR)
	mv $(APP_DIR).tgz $(APP_DIR)

uninstall:
	@rm -rf $(APP_DIR)

dvi:
	doxygen ./docs/Doxyfile
	$(OPEN) ./docs/html/index.html

tests: mostlyclean
	@$(CC) $(TEST_LIB) $(CHECKFLAGS) $(OPENMP) -o Test
	@./Test
	@rm -rf *.o *.a Test

gcov_report: mostlyclean
	@$(CC) $(TEST_LIB) -o Test $(GCOV) $(CHECKFLAGS)
	@./Test
	@lcov --no-external -c -d . -o $(APP).info
	@genhtml -o $(REPORTDIR) $(APP).info
	@$(OPEN) ./$(REPORTDIR)/index.html

check: style cppcheck leaks

style: 
	@clang-format -style=google -n -verbose */*.cc */*.h

cppcheck:
	@cppcheck $(CPPCHECKFLAGS) */*.cc */*.h

leaks: mostlyclean
	@$(CC) $(TEST_LIB) $(CHECKFLAGS) -o Test
	@$(CHECK_LEAKS) ./Test $(FILTER)
	@rm -rf *.o *.a Test

clean:
	@rm -rf *.o *.a *.out *.gcno *.gch *.gcda *.info *.tgz $(REPORTDIR) Test $(BUILD_DIR) $(APP_DIR) $(APP) ./docs/html

mostlyclean:
	@rm -rf *.o *.out *.gcno *.gch *.gcda *.info $(REPORTDIR) Test

open:
ifeq ($(OS), Linux)
	./$(APP_DIR)/$(APP)
else
	open $(APP_DIR)/$(APP).app
endif

my:
	@g++ ./model/my.cc ./model/utility/matrix_operations.cc $(OPENMP)  && ./a.out

speed:
	@$(CC) ./tests/speed_matrix_ops.cc ./model/utility/matrix_operations.cc $(OPENMP) && ./a.out

emnist:
	@$(CC) ./tests/parse_emnist_tests.cc ./model/io/io.cc && ./a.out
