def refresh_combo_box(comboBox, text):
    """Auxiliary function refreshing combo box with a given text.
    """
    idx = comboBox.findText(text)
    if idx != -1:
        comboBox.setCurrentIndex(idx)
        return True
    else:
        comboBox.setCurrentIndex(0)
        return False


# ----------------------------------------------------------------------
def colorizeWidget(widget, styleSheet, toolTip):
    """
    """
    widget.setStyleSheet(styleSheet)
    widget.setToolTip(toolTip)
