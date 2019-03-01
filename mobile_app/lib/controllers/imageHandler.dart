import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

final String BACKEND_URI =
    "http://10.0.0.22:8080/MalignantBirthmarkRecognition_war_exploded/analyze"; //TODO

Future<File> chooseCamera() async {
  return await ImagePicker.pickImage(source: ImageSource.camera);
}

Future<File> chooseGallery() async {
  return await ImagePicker.pickImage(source: ImageSource.gallery);
}

Future<String> analyze(File file) async {
  if (file == null) return "Error: No image selected.";

  String base64Image = base64Encode(file.readAsBytesSync());

  http.Response r = await http.post(
    BACKEND_URI,
    body: {
      "image": base64Image,
    },
  );

  print("STATUS: "+r.statusCode.toString());
  print("BODY: "+r.body.toString());
  return r.body.toString();
}
