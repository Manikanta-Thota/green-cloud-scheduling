PY ?= python3
JAVAC ?= javac
CP_SEP := $(if $(filter Windows_NT,$(OS)),;,:)
SRC_CORE := java/src/main/java/gcs/core/*.java
SRC_OPT := java/src/main/java/gcs/opt/*.java
MAVEN := https://repo1.maven.org/maven2

.PHONY: setup test test-py test-java build cloudsim-lib validate-cloudsim tune experiment sensitivity analysis demo clean

setup:
	$(PY) -m pip install -r requirements.txt

build:
	mkdir -p java/out && $(JAVAC) -d java/out $(SRC_CORE) $(SRC_OPT)

test: test-py test-java

test-py:
	cd python && $(PY) -m unittest discover -s tests -v

test-java: build
	java -cp java/out gcs.core.SelfCheck testdata/tiny_case.txt
	java -cp java/out gcs.core.SelfCheck testdata/parity_case.txt
	java -cp java/out gcs.core.SelfCheck testdata/parity_workflow_case.txt
	java -cp java/out gcs.opt.OptSelfCheck testdata/parity_workflow_case.txt

# CloudSim Plus 8.5.7 and its runtime dependencies, from Maven Central, into java/lib (not committed)
cloudsim-lib:
	mkdir -p java/lib && cd java/lib && for u in \
	  org/cloudsimplus/cloudsimplus/8.5.7/cloudsimplus-8.5.7.jar \
	  org/apache/commons/commons-math3/3.6.1/commons-math3-3.6.1.jar \
	  org/apache/commons/commons-lang3/3.17.0/commons-lang3-3.17.0.jar \
	  org/apache/commons/commons-collections4/4.4/commons-collections4-4.4.jar \
	  com/google/code/gson/gson/2.12.1/gson-2.12.1.jar \
	  ch/qos/logback/logback-core/1.5.18/logback-core-1.5.18.jar \
	  ch/qos/logback/logback-classic/1.5.18/logback-classic-1.5.18.jar \
	  org/slf4j/slf4j-api/2.0.17/slf4j-api-2.0.17.jar; do curl -sfO $(MAVEN)/$$u; done

validate-cloudsim: cloudsim-lib
	cd python && $(PY) -m gcs.validate_cloudsim

tune:
	cd python && $(PY) -m gcs.tune --algo ga && $(PY) -m gcs.tune --algo hpeas

experiment:
	cd python && $(PY) -m gcs.experiment --out ../results/main

sensitivity:
	cd python && $(PY) -m gcs.sensitivity --out ../results/sensitivity

analysis:
	cd python && $(PY) -m gcs.analysis --main ../results/main --out ../results/analysis --figures ../report/figures \
	  --front ../results/fronts/fronts --front-stem s76_r00
	cd python && $(PY) -m gcs.sensitivity --analyse --out ../results/sensitivity --figures ../report/figures

demo:
	cd python && $(PY) -m gcs.experiment --out ../results/demo --scenarios 4 --reps 3 --algos fcfs,minmin,hpeas

clean:
	rm -rf java/out java/out-sim python/gcs/__pycache__ python/tests/__pycache__
