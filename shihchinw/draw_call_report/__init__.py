import qrenderdoc as qrd
from typing import Optional
from . import util

class Window(qrd.CaptureViewer):
    def __init__(self, ctx: qrd.CaptureContext, version: str):
        super().__init__()

        self.mqt: qrd.MiniQtHelper = ctx.Extensions().GetMiniQtHelper()
        self.ctx = ctx
        self.version = version
        self.topWindow = self.mqt.CreateToplevelWidget('Draw Call Reporter', lambda c, w, d: window_closed())

        vert = self.mqt.CreateVerticalContainer()
        self.mqt.AddWidget(self.topWindow, vert)

        horz = self.mqt.CreateHorizontalContainer()

        label = self.mqt.CreateLabel()
        self.mqt.SetWidgetText(label, 'Export resources:')

        self.exportInputTexturesChkBox = self.mqt.CreateCheckbox(None)
        self.mqt.SetWidgetText(self.exportInputTexturesChkBox, 'Textures')

        self.exportOutputsChkBox = self.mqt.CreateCheckbox(None)
        self.mqt.SetWidgetText(self.exportOutputsChkBox, 'Render outputs')

        self.exportShaderChkBox = self.mqt.CreateCheckbox(None)
        self.mqt.SetWidgetText(self.exportShaderChkBox, 'Shaders')

        self.mqt.AddWidget(horz, label)
        self.mqt.AddWidget(horz, self.exportShaderChkBox)
        self.mqt.AddWidget(horz, self.exportInputTexturesChkBox)
        self.mqt.AddWidget(horz, self.exportOutputsChkBox)

        spacer = self.mqt.CreateSpacer(True)
        self.mqt.AddWidget(horz, spacer)
        self.mqt.AddWidget(vert, horz)

        horz = self.mqt.CreateHorizontalContainer()
        label = self.mqt.CreateLabel()
        self.mqt.SetWidgetText(label, 'Export draw count')

        self.drawCountSpinBox = self.mqt.CreateSpinbox(0, 1)
        self.mqt.SetSpinboxBounds(self.drawCountSpinBox, 1, 9999)
        self.mqt.SetSpinboxValue(self.drawCountSpinBox, 9999)
        self.mqt.AddWidget(horz, label)
        self.mqt.AddWidget(horz, self.drawCountSpinBox)
        spacer = self.mqt.CreateSpacer(True)
        self.mqt.AddWidget(horz, spacer)

        self.eventRangeChkBox = self.mqt.CreateCheckbox(None)
        self.mqt.SetWidgetText(self.eventRangeChkBox, 'Event range')
        self.mqt.AddWidget(horz, self.eventRangeChkBox)

        self.startEventSpinbox = self.mqt.CreateSpinbox(0, 1)
        self.mqt.SetSpinboxBounds(self.startEventSpinbox, 1, 99999)
        self.mqt.SetSpinboxValue(self.startEventSpinbox, 1)
        self.mqt.AddWidget(horz, self.startEventSpinbox)

        self.endEventSpinbox = self.mqt.CreateSpinbox(0, 1)
        self.mqt.SetSpinboxBounds(self.endEventSpinbox, 1, 99999)
        self.mqt.SetSpinboxValue(self.endEventSpinbox, 9999)
        self.mqt.AddWidget(horz, self.endEventSpinbox)

        spacer = self.mqt.CreateSpacer(True)
        self.mqt.AddWidget(horz, spacer)
        self.mqt.AddWidget(vert, horz)

        horz = self.mqt.CreateHorizontalContainer()
        self.exportBtn = self.mqt.CreateButton(self.OnExportReport)
        self.mqt.SetWidgetText(self.exportBtn, 'Export draw call report')
        self.mqt.AddWidget(horz, self.exportBtn)

        self.clampPixelRangeChkBox = self.mqt.CreateCheckbox(None)
        self.mqt.SetWidgetText(self.clampPixelRangeChkBox, 'Clamp Output Pixel Range')
        self.mqt.AddWidget(horz, self.clampPixelRangeChkBox)
        self.mqt.AddWidget(vert, horz)

        ctx.AddCaptureViewer(self)

    def OnDrawCountChanged(self, ctx, widget, text):
        print(f'Draw counts: {text}')

    def OnExportReport(self, ctx, widget, text):
        options = util.ExportOptions()

        if self.mqt.IsWidgetChecked(self.eventRangeChkBox):
            options.start_event_id = self.mqt.GetSpinboxValue(self.startEventSpinbox)
            options.end_event_id = self.mqt.GetSpinboxValue(self.endEventSpinbox)
            if options.end_event_id <= options.start_event_id:
                self.ctx.Extensions().ErrorDialog(f'End event id should be larger than start event id!')
                return

        options.output_dir = ctx.Extensions().OpenDirectoryName('Export draw call reporter')

        if not options.output_dir:
            return

        options.export_shaders = self.mqt.IsWidgetChecked(self.exportShaderChkBox)
        options.export_input_textures = self.mqt.IsWidgetChecked(self.exportInputTexturesChkBox)
        options.export_output_targets = self.mqt.IsWidgetChecked(self.exportOutputsChkBox)
        options.export_output_targets = self.mqt.IsWidgetChecked(self.exportOutputsChkBox)
        options.clamp_output_pixel_range = self.mqt.IsWidgetChecked(self.clampPixelRangeChkBox)
        options.draw_count = self.mqt.GetSpinboxValue(self.drawCountSpinBox)

        util.async_export(ctx, options)
        print(f'Dump resource to {options.output_dir}')

    def OnCaptureLoaded(self):
        pass

    def OnCaptureClosed(self):
        pass

    def OnSelectedEventChanged(self, event):
        pass

    def OnEventChanged(self, event):
        pass


_ext_version = ''
_cur_window: Optional[Window] = None

def window_closed():
    global _cur_window

    if _cur_window:
        _cur_window.ctx.RemoveCaptureViewer(_cur_window)

    _cur_window = None


def window_callback(ctx: qrd.CaptureContext, data):
    global _cur_window

    if not _cur_window:
        _cur_window = Window(ctx, _ext_version)
        if ctx.HasEventBrowser():
            ctx.AddDockWindow(_cur_window.topWindow, qrd.DockReference.TopOf, ctx.GetEventBrowser().Widget(), 0.1)
        else:
            ctx.AddDockWindow(_cur_window.topWindow, qrd.DockReference.MainToolArea, None)

    ctx.RaiseDockWindow(_cur_window.topWindow)


def register(version: str, ctx: qrd.CaptureContext):
    global _ext_version
    _ext_version = version

    ctx.Extensions().RegisterWindowMenu(qrd.WindowMenu.Window, ['Draw call report options'], window_callback)


def unregister():
    global _cur_window

    if _cur_window:
        # The window_closed() callback will unregister the capture viewer
        _cur_window.ctx.Extensions().GetMiniQtHelper().CloseToplevelWidget(_cur_window.topWindow)
        _cur_window = None