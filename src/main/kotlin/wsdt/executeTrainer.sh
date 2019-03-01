#!/usr/bin/env bash

javac ./ml/train/Trainer.kt
java -Xms1G -Xmx8G -Dorg.bytedeco.javacpp.maxbytes=8G -Dorg.bytedeco.javacpp.maxphysicalbytes=10G ./ml/train/Trainer