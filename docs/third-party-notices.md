# Third-party components

The project source is licensed under MIT as stated in `LICENSE`. This archive contains source, tests, generated examples, and documentation. It does not redistribute installed dependencies, a Python runtime, or model weights. Installing dependencies does not make their licenses identical to this project's license.

## Desktop and scientific stack

The project uses NumPy, pandas, Matplotlib, ReportLab, Qt for Python (PySide6 Essentials), and OpenCV. Their distributions carry their respective license/notice files. Retain and review those files if you build a binary distribution containing them.

The desktop extra uses the smaller PySide6 Essentials package, which includes the Qt Core, Gui, and Widgets modules used here. See [Qt's package listing](https://github.com/pyside/pyside-setup/blob/dev/README.pyside6_essentials.md) and [Qt for Python licensing](https://doc.qt.io/qtforpython-6/licenses.html).

PDF generation registers the DejaVu font supplied within the installed Matplotlib distribution; no separate font asset is copied into this source tree.

## Optional YOLO

Ultralytics and model weights are optional. The default color-marker workflow does not import them. Ultralytics documents AGPL-3.0 and Enterprise licensing options; consult its [current license terms](https://www.ultralytics.com/license) for the intended distribution and deployment. The MIT notice on this repository does not override those terms.

Weights may be downloaded by the optional detector when a model name is used. Only load trusted weights. This release does not bundle a model, claim to grant rights to a third-party model, or verify a particular GPU/driver combination.

## Included examples

The synthetic coordinate files and colored-marker video are generated from equations and drawing primitives by this project. The archive subfolder preserves unverified data from the supplied earlier version; its numerical outputs are not validation references.
