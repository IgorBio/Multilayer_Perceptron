.PHONY: all build rebuild install uninstall run dist dvi tests clean cppcheck style leaks gcov_report train emnist speed

APP=MultilayerPerceptron
APP_DIR=../$(APP)
BUILD_DIR=../build
TEST_BUILD_DIR=$(BUILD_DIR)/tests
OS=$(shell uname)

ifeq ($(OS), Linux)
	CHECK_LEAKS=CK_FORK=no valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --log-file=valgrind.log
	OPEN=xdg-open
	DIR=/$(APP)
	RUN_APP=./$(APP_DIR)/$(APP)
else
	CHECK_LEAKS=CK_FORK=no leaks --atExit --
	FILTER=--gtest_filter=-*.Exception*
	OPEN=open
	DIR=/$(APP).app
	RUN_APP=open $(APP_DIR)/$(APP).app
endif

all: build

build:
	@cmake -S . -B $(BUILD_DIR)
	@cmake --build $(BUILD_DIR)

rebuild: clean build

install: build uninstall
	@mkdir -p $(APP_DIR)
	@cp -r $(BUILD_DIR)$(DIR) $(APP_DIR)

run: install
	$(RUN_APP)

unzip:
	tar -xvzf ../datasets/emnist-letters.tar -C ../datasets/

dist: install
	tar -czvf $(APP_DIR).tgz $(APP_DIR)
	mv $(APP_DIR).tgz $(APP_DIR)

dvi:
	@mkdir -p ../build/docs
	doxygen ./docs/Doxyfile
	$(OPEN) $(BUILD_DIR)/docs/html/index.html

uninstall:
	@rm -rf $(APP_DIR)

tests:
	@cmake -S ./tests -B $(TEST_BUILD_DIR)
	@cmake --build $(TEST_BUILD_DIR)
	@$(TEST_BUILD_DIR)/Tests

check: style cppcheck leaks

style: 	
	@clang-format -style=google -n -verbose */*.cc  */*.h */*/*.cc  */*/*.h

cppcheck: build
	@cd $(BUILD_DIR); make cppcheck

leaks: tests
	@$(CHECK_LEAKS) $(TEST_BUILD_DIR)/Tests $(FILTER)

clean:
	@rm -rf $(BUILD_DIR) *.log

train:
	@cmake -S . -B $(BUILD_DIR)
	@cmake --build $(BUILD_DIR) --target Training
	@$(BUILD_DIR)/Training

emnist:
	@cmake -S ./tests -B $(TEST_BUILD_DIR)
	@cmake --build $(TEST_BUILD_DIR) --target Emnist
	@$(TEST_BUILD_DIR)/Emnist

speed:
	@cmake -S ./tests -B $(TEST_BUILD_DIR)
	@cmake --build $(TEST_BUILD_DIR) --target Speed
	@$(TEST_BUILD_DIR)/Speed
