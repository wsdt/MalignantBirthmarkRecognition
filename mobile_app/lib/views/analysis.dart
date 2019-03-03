import 'dart:io';
import "dart:convert";

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:mobile_app/controllers/imageHandler.dart' as ih;

class MyHomePage extends StatefulWidget {
  MyHomePage({Key key}) : super(key: key);

  @override
  State<StatefulWidget> createState() {
    return MyHomePageState();
  }
}

class MyHomePageState extends State<MyHomePage> {
  File file;
  int precision = 6;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Analysis"),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                CupertinoButton(
                  child: Text("Make picture"),
                  onPressed: () async {
                    var f = await ih.chooseCamera();
                    setState(() {
                      file = f;
                    });
                  },
                ),
              ],
            ),
            Row(
              children: [
                CupertinoButton(
                  onPressed: () async {
                    var f = await ih.chooseGallery();
                    setState(() {
                      file = f;
                    });
                  },
                  child: Text("Choose from gallery"),
                ),
                SizedBox(width: 10.0),
                CupertinoButton(
                  onPressed: () async {
                    var result = await jsonDecode(await ih.analyze(file));

                    showDialog(
                      context: context,
                      builder: (BuildContext ctx) => CupertinoAlertDialog(
                            title: const Text("Results"),
                            content: Column(
                              children: <Widget>[
                                Row(children: <Widget>[
                                  Text(result[0]["name"] +
                                      ": " +
                                  this.formatPrecision(result[0]["confidence"]))
                                ]),
                                Row(children: <Widget>[
                                  Text(result[1]["name"] +
                                      ": " +
                                      this.formatPrecision(result[1]["confidence"]))
                                ]),
                                Row(children: <Widget>[
                                  Text(result[2]["name"] +
                                      ": " +
                                      this.formatPrecision(result[2]["confidence"]))
                                ]),
                              ],
                            ),
                          ),
                    );
                  },
                  child: Text("Analyze now"),
                ),
              ],
            ),
            file == null ? Text("No image selected") : Image.file(file)
          ],
        ),
      ),
    );
  }

  String formatPrecision(String rawValue) {
    return (double.parse(rawValue)*100).toStringAsFixed(precision)+" %";
  }
}
