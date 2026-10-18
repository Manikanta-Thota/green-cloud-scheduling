PY ?= python3
JAVAC ?= javac

.PHONY: setup test test-py test-java demo clean
setup:
	$(PY) -m pip install -r requirements.txt

test: test-py test-java

test-py:
	cd python && $(PY) -m unittest discover -s tests -v

test-java:
	mkdir -p java/out && $(JAVAC) -d java/out java/src/main/java/gcs/core/*.java
	java -cp java/out gcs.core.SelfCheck testdata/tiny_case.txt
	java -cp java/out gcs.core.SelfCheck testdata/parity_case.txt

demo:
	cd python && $(PY) -m gcs.run_baselines --scenario 0 --reps 3 --out ../results/demo_baselines.csv

clean:
	rm -rf java/out python/gcs/__pycache__ python/tests/__pycache__
