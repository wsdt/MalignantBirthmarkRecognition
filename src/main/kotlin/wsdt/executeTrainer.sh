#!/usr/bin/env bash

javac ./ml/train/Trainer.kt
java -Xms2G -Xmx14G -Dorg.bytedeco.javacpp.maxbytes=14G -Dorg.bytedeco.javacpp.maxphysicalbytes=35G ./ml/train/Trainer