# Command example: <command> M=<*.tflite> B=200

ifndef M
$(error ERROR! Please, provide your model file.)
endif

ifndef B
override B = 200
endif

run_classifier:
	python3 gtsrb_tflite_classifier.py -m ${M} -b ${B}
