import qrenderdoc as qrd
import renderdoc as rd

import csv
import os
import re
import warnings

from datetime import datetime


_image_ext_dict = {
	rd.FileType.DDS: 'dds',
	rd.FileType.PNG: 'png',
	rd.FileType.JPG: 'jpg',
	rd.FileType.BMP: 'bmp',
	rd.FileType.TGA: 'tga',
	rd.FileType.HDR: 'hdr',
	rd.FileType.EXR: 'exr' }


class ExportOptions:

	def __init__(self) -> None:
		self.draw_count = 999
		self.start_event_id = 0
		self.end_event_id = -1
		self.output_dir = None
		self.export_input_textures = False
		self.export_output_targets = False
		self.export_shaders = False
		self.force_overwrite = False


def _get_first_action(controller: rd.ReplayController):
	# Start iterating from the first real action as a child of markers
	action = controller.GetRootActions()[0]

	while len(action.children) > 0:
		action = action.children[0]

	return action


def _get_shader_resource_id(shader):
	return shader.shaderResourceId if type(shader) == rd.GLShader else shader.resourceId


def _get_depth_function_desc(depth_state):
	if not depth_state.depthEnable:
		return ''

	compare_func_str = str(depth_state.depthFunction)
	return re.match(r'CompareFunction[.](\w+)', compare_func_str).group(1)


def _save_texture(resourceId, controller, filepath, image_type):
	if resourceId == rd.ResourceId.Null():
		return False

	filepath = f'{filepath}.{_image_ext_dict[image_type]}'
	if os.path.exists(filepath):
		return False

	folderpath = os.path.dirname(filepath)
	if not os.path.exists(folderpath):
		os.makedirs(folderpath)

	texsave = rd.TextureSave()
	texsave.resourceId = resourceId
	texsave.alpha = rd.AlphaMapping.BlendToCheckerboard

	# Only store the first mip and slice.
	texsave.mip = 0
	texsave.slice.sliceIndex = 0

	# Choose file type according to render target properties.
	texsave.destType = image_type
	controller.SaveTexture(texsave, filepath)
	return True


def _get_image_type_from_tex_desc(texture_desc):
	if texture_desc.cubemap:
		return rd.FileType.DDS

	comp_type = texture_desc.format.compType
	if comp_type == rd.CompType.Float or comp_type == rd.CompType.Depth:
		return rd.FileType.EXR

	return rd.FileType.PNG


class DrawStateExtractor:

	def __init__(self, pipe_state, capture, controller, clamp_pixel_range=False):
		self.pipe_state = pipe_state
		self.capture = capture
		self.controller = controller
		self.vert_shader_id = _get_shader_resource_id(pipe_state.vertexShader)
		self.frag_shader_id = _get_shader_resource_id(pipe_state.fragmentShader)
		self.comp_shader_id = _get_shader_resource_id(pipe_state.computeShader)
		self.clamp_pixel_range = clamp_pixel_range

	def _get_image_type(self, resource_id):
		texture_desc = self.capture.GetTexture(resource_id)
		if self.clamp_pixel_range:
			return rd.FileType.PNG

		return _get_image_type_from_tex_desc(texture_desc)

	def _get_texture_info(self, resource_id):
		resource_desc = self.capture.GetResource(resource_id)
		texture_desc = self.capture.GetTexture(resource_id)
		return (resource_desc.name, texture_desc)

	def _get_shader_content(self, shader):
		reflection = shader.reflection
		if not reflection:
			return None

		encoding = reflection.encoding
		if encoding == rd.ShaderEncoding.GLSL:
			return reflection.rawBytes.decode('utf-8').rstrip('\x00')
		elif encoding == rd.ShaderEncoding.SPIRV:
			state = self.controller.GetPipelineState()
			pipe = state.GetGraphicsPipelineObject()
			return self.controller.DisassembleShader(pipe, reflection, '')

		raise NotImplementedError(f'Unsupported shader encoding {reflection.encoding}')


	def _save_shader(self, shader, output_dir):
		if not shader:
			raise RuntimeError('Can not find shader instance.')

		if shader.stage == rd.ShaderStage.Vertex:
			shader_type_name = 'vert'
		elif shader.stage == rd.ShaderStage.Fragment:
			shader_type_name = 'frag'
		elif shader.stage == rd.ShaderStage.Compute:
			shader_type_name = 'comp'
		else:
			raise NotImplemented(f'Unsupported shader type: {shader.stage}')

		shader_id = int(_get_shader_resource_id(shader))
		if shader_id == 0:
			return

		filepath = os.path.join(output_dir, f'Shader-{shader_id:03d}.{shader_type_name}')
		if os.path.exists(filepath):
			# Skip duplicated shaders. One shader could be used in multiple programs.
			return

		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		content = self._get_shader_content(shader)
		if not content:
			warnings.warn(f'Cannot find shader content {shader_id}')
			return

		with open(filepath, 'w', encoding='utf-8') as out_file:
			out_file.writelines(content)
			print(f'\tWrite shader {shader_id} to {filepath}')

	def save_shaders(self, output_dir):
		self._save_shader(self.pipe_state.computeShader, output_dir)
		self._save_shader(self.pipe_state.vertexShader, output_dir)
		self._save_shader(self.pipe_state.fragmentShader, output_dir)


class GLDrawStateExtractor(DrawStateExtractor):

	def __init__(self, pipe_state, capture, clamp_pixel_range=False):
		super().__init__(pipe_state, capture, clamp_pixel_range)

	def save_input_textures(self, event_id, controller, output_dir):
		for texture in self.pipe_state.textures:
			resource_id = texture.resourceId
			if resource_id == rd.ResourceId.Null():
				continue

			if texture.type != rd.TextureType.Texture2D:
				warnings.warn(f'[{event_id}] Non-supported texture type for dump: {resource_id}')
				continue

			resource_desc = self.capture.GetResource(resource_id)
			filepath = os.path.join(output_dir, f'{resource_desc.name}')
			if _save_texture(resource_id, controller, filepath, self._get_image_type(resource_id)):
				print(f'└ Save input texture: {filepath}')

	def save_output_targets(self, controller, event_id, output_dir):
		draw_fbo = self.pipe_state.framebuffer.drawFBO

		for idx, attachment in enumerate(draw_fbo.colorAttachments):
			resource_id = attachment.resourceId
			if resource_id == rd.ResourceId.Null():
				continue

			filepath = os.path.join(output_dir, f'draw_color_{event_id:04d}_{idx}')
			if _save_texture(resource_id, controller, filepath, self._get_image_type(resource_id)):
				print(f'└ Save output target {filepath}')

		filepath = os.path.join(output_dir, f'draw_depth_{event_id:04d}_{idx}')
		if _save_texture(draw_fbo.depthAttachment.resourceId, controller, filepath, rd.FileType.PNG):
			print(f'└ Save output target {filepath}')

	def get_input_texture_desc_map(self):
		vs_tex_bind_points = set()
		vertex_shader = self.pipe_state.vertexShader
		if vertex_shader.reflection:
			for resource in vertex_shader.reflection.readOnlyResources:
				if resource.isTexture:
					vs_tex_bind_points.add(resource.bindPoint)

		tex_desc_map = { 'VS': {}, 'FS': {} }
		for idx, texture in enumerate(self.pipe_state.textures):
			resource_id = texture.resourceId
			if resource_id == rd.ResourceId.Null():
				continue

			name, tex_desc = self._get_texture_info(resource_id)
			category = 'VS' if idx in vs_tex_bind_points else 'FS'
			tex_desc_map[category][name] = tex_desc

		return tex_desc_map

	def get_output_desc_map(self, action):
		tex_desc_map = {}
		for idx, resource_id in enumerate(action.outputs):
			if resource_id == rd.ResourceId.Null():
				continue

			name, tex_desc = self._get_texture_info(resource_id)
			if tex_desc:
				tex_desc_map[name] = tex_desc

		return tex_desc_map

	def get_fbo_id(self):
		return int(self.pipe_state.framebuffer.drawFBO.resourceId)

	def get_viewport_info(self):
		for viewport in self.pipe_state.rasterizer.viewports:
			if viewport.enabled:
				return f'{int(viewport.width)} x {int(viewport.height)}'

	def _get_color_attachment_count(self):
		framebuffer = self.pipe_state.framebuffer
		count = 0
		for attachment in framebuffer.drawFBO.colorAttachments:
			resource_id = attachment.resourceId
			if resource_id != rd.ResourceId.Null():
				count += 1
		return count

	def get_color_write_masks(self):
		results = []
		blends = self.pipe_state.framebuffer.blendState.blends
		color_attachment_count = self._get_color_attachment_count()
		for idx in range(color_attachment_count):
			blend = blends[idx]
			mask = blend.writeMask
			channels = ['R', 'G', 'B', 'A']
			writeMaskTokens = []
			for idx, ch in enumerate(channels):
				token = ch if mask & (1 << idx) else '_'
				writeMaskTokens.append(token)

			results.append(''.join(writeMaskTokens))

		return results

	def append_draw_states(self, action, action_name, csv_writer):
		draw_states = [action.eventId, action_name, action.numIndices, action.numInstances,
					   self.get_viewport_info(), self.get_fbo_id(),
					   int(self.vert_shader_id),
					   int(self.frag_shader_id),
					   int(self.comp_shader_id)]

		for category, tex_dict in self.get_input_texture_desc_map().items():
			input_tex_descs = []
			for name, desc in tex_dict.items():
				info = f'{name}: {desc.width} x {desc.height}, {desc.format.Name()}'
				input_tex_descs.append(info)
			draw_states.append('\n'.join(input_tex_descs))

		output_tex_descs = []
		for name, desc in self.get_output_desc_map(action).items():
			info = f'{name}: {desc.width} x {desc.height}, {desc.format.Name()}'
			output_tex_descs.append(info)
		draw_states.append('\n'.join(output_tex_descs))

		write_masks = self.get_color_write_masks()
		draw_states.append('\n'.join(write_masks))

		depth_state = self.pipe_state.depthState
		draw_states.append(_get_depth_function_desc(depth_state))
		draw_states.append('V' if depth_state.depthWrites else '')
		csv_writer.writerow(draw_states)


class VKDrawStateExtractor(DrawStateExtractor):

	def __init__(self, pipe_state, capture, clamp_pixel_range=False):
		super().__init__(pipe_state, capture, clamp_pixel_range)



def export_gl_action(capture, state, action, controller, options: ExportOptions, csv_writer=None):
	print(f'Extracting draw [EID: {action.eventId:05d}]')

	draw_state_extractor = GLDrawStateExtractor(state, capture, controller)
	output_dir = options.output_dir

	if options.export_shaders:
		draw_state_extractor.save_shaders(os.path.join(output_dir, 'shaders'))

	if options.export_input_textures:
		draw_state_extractor.save_input_textures(action.eventId, controller, os.path.join(output_dir, 'textures'))

	if options.export_output_targets:
		draw_state_extractor.save_output_targets(controller, action.eventId, os.path.join(output_dir, 'outputs'))

	if csv_writer:
		action_name = action.GetName(controller.GetStructuredFile())
		draw_state_extractor.append_draw_states(action, action_name, csv_writer)


def traverse_draw_action(action: rd.ActionDescription, count: int, start_event_id: int, end_event_id: int):
	while action and count > 0:
		if action.eventId <= start_event_id:
			action = action.next
			continue
		elif end_event_id > 0 and action.eventId >= end_event_id:
			raise StopIteration

		if action.flags & rd.ActionFlags.Drawcall:
			yield action
			count -= 1
		action = action.next


def _get_time_token():
	return datetime.now().strftime('%Y%m%d_%H%M%S')


def export_draw_call_states(controller: rd.ReplayController, capture: qrd.CaptureContext, options: ExportOptions):

	capture_filepath = capture.GetCaptureFilename()
	if not options.output_dir:
		options.output_dir = os.path.dirname(capture_filepath)

	capture_filename = os.path.basename(capture_filepath)
	basename, _ = os.path.splitext(capture_filename)
	time_token = _get_time_token()
	report_csv_path = os.path.join(options.output_dir, f'{basename}_{time_token}.csv')

	header = ['Event ID', 'Name', 'Vertex Count', 'Instance Count',
			  'Viewport', 'Framebuffer',
			  'Vert-Shader', 'Frag-Shader', 'Compute Shader',
			  'VS Textures', 'FS Textures',
			  'Outputs', 'Color Write Mask', 'Depth State', 'Depth Write']

	if not os.path.exists(options.output_dir):
		os.makedirs(options.output_dir)

	with open(report_csv_path, 'w', newline='') as csvfile:
		csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow(header)

		action = _get_first_action(controller)
		for action in traverse_draw_action(action, options.draw_count, options.start_event_id, options.end_event_id):
			controller.SetFrameEvent(action.eventId, True)
			state = controller.GetPipelineState()
			if state.IsCaptureGL():
				gl_state = controller.GetGLPipelineState()
				export_gl_action(capture, gl_state, action, controller, options, csv_writer)
			elif state.IsCaptureVK():
				raise NotImplementedError('Not support Vulkan yet.')

	print(f'Finish exporting {capture_filename}')


def async_export(ctx: qrd.CaptureContext, options: ExportOptions):
	def _replay_callback(r: rd.ReplayController):
		export_draw_call_states(r, ctx, options)
	ctx.Replay().AsyncInvoke('DrawCallReporter', _replay_callback)


if 'pyrenderdoc' in globals():
	options = ExportOptions()
	options.draw_count = 5
	pyrenderdoc.Replay().BlockInvoke(lambda ctx: export_draw_call_states(ctx, pyrenderdoc, options))
