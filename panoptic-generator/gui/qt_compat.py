"""Qt compatibility layer — supports PyQt6 or PyQt5."""

try:
    from PyQt6.QtCore import *  # noqa: F401, F403
    from PyQt6.QtGui import *  # noqa: F401, F403
    from PyQt6.QtWidgets import *  # noqa: F401, F403
    from PyQt6.QtCore import Qt, pyqtSignal as Signal

    QT_VERSION = 6
except ImportError:
    from PyQt5.QtCore import *  # noqa: F401, F403
    from PyQt5.QtGui import *  # noqa: F401, F403
    from PyQt5.QtWidgets import *  # noqa: F401, F403
    from PyQt5.QtCore import Qt, pyqtSignal as Signal

    QT_VERSION = 5
