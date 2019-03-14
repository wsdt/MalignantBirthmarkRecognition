#!/usr/bin/env bash

# Shell commands to start trainer. Used to enlarge maximum bytes for java vm to avoid OutOfMemoryExceptions.
javac ./ml/train/Trainer.kt
java -Xms2G -Xmx14G -Dorg.bytedeco.javacpp.maxbytes=14G -Dorg.bytedeco.javacpp.maxphysicalbytes=35G ./ml/train/Trainer