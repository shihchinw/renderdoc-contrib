import qrenderdoc as qrd
import renderdoc as rd

import csv
import os
import re
import subprocess as sp
import tempfile
import warnings

from datetime import datetime

_VERBOSE = False
_SPIRV_CROSS_PATH = "C:/Program Files/RenderDoc/plugins/spirv/spirv-cross.exe"

_IMAGE_EXT_NAME_MAP = {
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
		self.clamp_output_pixel_range = False
		self.force_overwrite = False
		self.capture_gpu_duration = True


_DRAW_STATE_LABEL_MAP = {
	'event_id' : 'Event ID',
	'custom_name' : 'Custom Name',
	'api_name' : 'Name',
	'vertex_count' : 'Vertex Count',
	'instance_count' : 'Instance Count',
	'dispatch_dimension' : 'Dispatch Dimension',
	'elapsed_time' : 'GPU Duration (ms)',
	'viewport_size' : 'Viewport',
	'pass_switch' : 'Pass Switch',		# Switch to another frame buffer or dispatch compute task
	'renderpass_id' : 'Render Pass',
	'vert_shader_id' : 'Vertex Shader',
	'frag_shader_id' : 'Fragment Shader',
	'comp_shader_id' : 'Compute Shader',
	'vs_textures' : 'VS Textures',
	'fs_textures' : 'FS Textures',
	'output_targets' : 'Outputs',
	'color_mask' : 'Color Write',
	'depth_state' : 'Depth State',
	'depth_write' : 'Depth Write'
}


class DrawCallState:

	def __init__(self, action, name, elapsed_time) -> None:
		self.event_id = action.eventId
		self.api_name = name
		self.custom_name = '' if not action.parent else action.parent.customName
		self.elapsed_time = '' if not elapsed_time else elapsed_time

		if action.flags & rd.ActionFlags.Drawcall:
			self.vertex_count = action.numIndices
			self.instance_count = action.numInstances
			self.dispatch_dimension = None
			self.pass_switch = ''
		elif action.flags & rd.ActionFlags.Dispatch:
			self.vertex_count = None
			self.instance_count = None
			self.dispatch_dimension = action.dispatchDimension
			self.pass_switch = 'v' # Treat each compute dispatch a new pass item.

		self.viewport_size = None
		self.renderpass_id = 0
		self.vert_shader_id = 0
		self.frag_shader_id = 0
		self.comp_shader_id = 0
		self.vs_textures = None
		self.fs_textures = None
		self.output_targets = None
		self.color_mask = None
		self.depth_state = None
		self.depth_write = False

	def write_to_csv_dict(self, csv_writer):
		csv_writer.writerow(self.__dict__)


def _get_first_action(controller: rd.ReplayController):
	# Start iterating from the first real action as a child of markers
	action = controller.GetRootActions()[0]

	while len(action.children) > 0:
		action = action.children[0]

	return action


def _get_shader_resource_id(shader):
	return shader.shaderResourceId if type(shader) == rd.GLShader else shader.resourceId


def _get_depth_function_desc(depth_state):
	if (not getattr(depth_state, 'depthEnable', True) or
		not getattr(depth_state, 'depthTestEnable', True)):
		return ''

	compare_func_str = str(depth_state.depthFunction)
	return re.match(r'CompareFunction[.](\w+)', compare_func_str).group(1)


def _save_texture(resourceId, controller, filepath, image_type):
	if resourceId == rd.ResourceId.Null():
		return False

	filepath = f'{filepath}.{_IMAGE_EXT_NAME_MAP[image_type]}'
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


class DrawStateExtractor:

	def __init__(self, capture, controller, action):
		self.capture = capture
		self.controller = controller
		self.action = action

	def _get_resource_image_type(self, resource_id, clamp_pixel_range):
		texture_desc = self.capture.GetTexture(resource_id)
		if texture_desc.cubemap:
			return rd.FileType.DDS

		if clamp_pixel_range:
			return rd.FileType.PNG

		comp_type = texture_desc.format.compType
		if comp_type == rd.CompType.Float or comp_type == rd.CompType.Depth:
			return rd.FileType.EXR

		return rd.FileType.PNG

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
			# If the shader is SPIRV, we first dump the byte code to a temp file,
			# then use spirv-cross for readable source code conversion.
			tmp_file_path = os.path.join(tempfile.gettempdir(), 'spirv.bytes')
			with open(tmp_file_path, 'wb') as f:
				f.write(reflection.rawBytes)
				f.close()
				cmd = f'"{_SPIRV_CROSS_PATH}" -V {tmp_file_path}'
				return sp.check_output(cmd, stderr=sp.STDOUT, shell=True, encoding='utf-8')

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
			out_file.write(content)
			print(f'└ Write shader {shader_id} to {filepath}')

	def save_shaders(self, output_dir):
		self._save_shader(self.pipe_state.computeShader, output_dir)
		self._save_shader(self.pipe_state.vertexShader, output_dir)
		self._save_shader(self.pipe_state.fragmentShader, output_dir)

	def extract_resources(self, options: ExportOptions):
		output_dir = options.output_dir

		if options.export_shaders:
			self.save_shaders(os.path.join(output_dir, 'shaders'))

		event_id = self.action.eventId
		if options.export_input_textures:
			self.save_input_textures(os.path.join(output_dir, 'textures'))

		if options.export_output_targets:
			self.save_output_targets(os.path.join(output_dir, 'outputs'), options.clamp_output_pixel_range)

	def write_summary(self, action_elapsed_time, csv_writer):
		assert(csv_writer != None)
		action_name = self.action.GetName(self.controller.GetStructuredFile())
		self._append_draw_state_summary(action_name, action_elapsed_time, csv_writer)


class GLDrawStateExtractor(DrawStateExtractor):

	last_fbo_id = None

	def __init__(self, capture, controller, action):
		super().__init__(capture, controller, action)
		self.pipe_state = controller.GetGLPipelineState()

	def save_input_textures(self, output_dir, clamp_pixel_range=False):
		for texture in self.pipe_state.textures:
			resource_id = texture.resourceId
			if resource_id == rd.ResourceId.Null():
				continue

			if texture.type != rd.TextureType.Texture2D:
				event_id = self.action.eventId
				warnings.warn(f'[{event_id}] Non-supported texture type for dump: {resource_id}')
				continue

			resource_desc = self.capture.GetResource(resource_id)
			filepath = os.path.join(output_dir, f'{resource_desc.name}')
			image_type = self._get_resource_image_type(resource_id, clamp_pixel_range)
			if _save_texture(resource_id, self.controller, filepath, image_type):
				print(f'└ Save input texture: {filepath}')

	def save_output_targets(self, output_dir, clamp_pixel_range=False):
		draw_fbo = self.pipe_state.framebuffer.drawFBO
		event_id = self.action.eventId

		for idx, attachment in enumerate(draw_fbo.colorAttachments):
			resource_id = attachment.resourceId
			if resource_id == rd.ResourceId.Null():
				continue

			filepath = os.path.join(output_dir, f'draw_color_{event_id:04d}_{idx}')
			image_type = self._get_resource_image_type(resource_id, clamp_pixel_range)
			if _save_texture(resource_id, self.controller, filepath, image_type):
				print(f'└ Save output target {filepath}')

		resource_id = draw_fbo.depthAttachment.resourceId
		if resource_id == rd.ResourceId.Null():
			return

		filepath = os.path.join(output_dir, f'draw_depth_{event_id:04d}')
		image_type = self._get_resource_image_type(resource_id, clamp_pixel_range)
		if _save_texture(resource_id, self.controller, filepath, image_type):
			print(f'└ Save output target {filepath}')

	def _get_shader_read_tex_bindpoints(self, shader: rd.GLShader):
		""" Return bindpoint index set of shader's read textures.

		Returns:
			The set contains corresponding bindpoint indices of read textures.
		"""
		shader_reflection = shader.reflection
		bindpoint_set = set()
		if not shader_reflection:
			return bindpoint_set

		bindpoint_mapping = shader.bindpointMapping
		for resource in shader_reflection.readOnlyResources:
			if not resource.isTexture:
				continue

			bindpoint = bindpoint_mapping.readOnlyResources[resource.bindPoint] # Need remapping indices to access texture array.
			bindpoint_set.add(bindpoint.bind)

		return bindpoint_set

	def get_input_texture_desc_map(self):
		vs_tex_bind_points = self._get_shader_read_tex_bindpoints(self.pipe_state.vertexShader)
		fs_tex_bind_points = self._get_shader_read_tex_bindpoints(self.pipe_state.fragmentShader)

		tex_desc_map = { 'VS': {}, 'FS': {} }
		for idx, texture in enumerate(self.pipe_state.textures):
			resource_id = texture.resourceId
			if resource_id == rd.ResourceId.Null():
				continue

			name, tex_desc = self._get_texture_info(resource_id)
			if idx in vs_tex_bind_points:
				tex_desc_map['VS'][name] = tex_desc
			elif idx in fs_tex_bind_points:
				tex_desc_map['FS'][name] = tex_desc

		return tex_desc_map

	def get_output_desc_map(self, action: rd.ActionDescription):
		tex_desc_map = {}

		# Warning action.outputs would return null resources when there is no
		# target blends. Therefore, we retrieve output targets from draw FBO instead.
		draw_fbo = self.pipe_state.framebuffer.drawFBO
		attachments = draw_fbo.colorAttachments + [draw_fbo.depthAttachment]
		for attachment in attachments:
			resource_id = attachment.resourceId
			if resource_id == rd.ResourceId.Null():
				continue

			name, tex_desc = self._get_texture_info(resource_id)
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

	def _append_draw_state_summary(self, action_name, action_elapsed_time, csv_writer):
		draw_state = DrawCallState(self.action, action_name, action_elapsed_time)
		draw_state.viewport_size = self.get_viewport_info()
		draw_state.renderpass_id = self.get_fbo_id()

		if draw_state.renderpass_id != GLDrawStateExtractor.last_fbo_id:
			draw_state.pass_switch = 'v'
		GLDrawStateExtractor.last_fbo_id = draw_state.renderpass_id

		draw_state.vert_shader_id = int(_get_shader_resource_id(self.pipe_state.vertexShader))
		draw_state.frag_shader_id = int(_get_shader_resource_id(self.pipe_state.fragmentShader))
		draw_state.comp_shader_id = int(_get_shader_resource_id(self.pipe_state.computeShader))

		for category, tex_dict in self.get_input_texture_desc_map().items():
			input_tex_descs = []
			for name, desc in tex_dict.items():
				info = f'{name}: {desc.width} x {desc.height}, {desc.format.Name()}'
				input_tex_descs.append(info)

			input_tex_info_str = '\n'.join(input_tex_descs)
			if category == 'VS':
				draw_state.vs_textures = input_tex_info_str
			else:
				draw_state.fs_textures = input_tex_info_str

		output_tex_descs = []
		for name, desc in self.get_output_desc_map(self.action).items():
			info = f'{name}: {desc.width} x {desc.height}, {desc.format.Name()}'
			output_tex_descs.append(info)
		draw_state.output_targets = '\n'.join(output_tex_descs)

		write_masks = self.get_color_write_masks()
		draw_state.color_mask = '\n'.join(write_masks)

		depth_state = self.pipe_state.depthState
		draw_state.depth_state = _get_depth_function_desc(depth_state)
		draw_state.depth_write = 'v' if depth_state.depthWrites else ''
		draw_state.write_to_csv_dict(csv_writer)


class VKDrawStateExtractor(DrawStateExtractor):

	last_render_pass_id = None

	def __init__(self, capture, controller, action):
		super().__init__(capture, controller, action)
		self.pipe_state = controller.GetVulkanPipelineState()

	def _get_shader_read_tex_resource_ids(self, shader_stage):
		pipe = self.controller.GetPipelineState()

		resource_ids = []
		for bound_array in pipe.GetReadOnlyResources(shader_stage, True):
			for bound_resource in bound_array.resources:
				resource_id = bound_resource.resourceId
				if resource_id == rd.ResourceId.Null():
					continue

				resource_desc = self.capture.GetResource(resource_id)
				if resource_desc.type == rd.ResourceType.Texture:
					resource_ids.append(resource_id)

		return resource_ids

	def _get_active_read_tex_resource_ids(self):
		resource_ids = []
		resource_ids.extend(self._get_shader_read_tex_resource_ids(rd.ShaderStage.Vertex))
		resource_ids.extend(self._get_shader_read_tex_resource_ids(rd.ShaderStage.Fragment))
		resource_ids.extend(self._get_shader_read_tex_resource_ids(rd.ShaderStage.Compute))
		return resource_ids

	def save_input_textures(self, output_dir, clamp_pixel_range=False):
		for resource_id in self._get_active_read_tex_resource_ids():
			texture_desc = self.capture.GetTexture(resource_id)
			if texture_desc.dimension != 2:
				event_id = self.action.eventId
				warnings.warn(f'[{event_id}] Non-supported texture type for dump: {resource_id}')
				continue

			resource_desc = self.capture.GetResource(resource_id)
			filepath = os.path.join(output_dir, f'{resource_desc.name}')
			image_type = self._get_resource_image_type(resource_id, clamp_pixel_range)
			if _save_texture(resource_id, self.controller, filepath, image_type):
				print(f'└ Save input texture: {filepath}')

	def _get_output_target_resource_ids(self):
		current_pass = self.pipe_state.currentPass
		attachments = current_pass.framebuffer.attachments
		attachment_count = len(attachments)
		resource_ids = []

		for idx in current_pass.renderpass.colorAttachments:
			if not (0 <= idx < attachment_count):
				# Skip invalid attachment index
				continue

			resource_id = attachments[idx].imageResourceId
			if resource_id != rd.ResourceId.Null():
				resource_ids.append(resource_id)

		depth_attach_idx = current_pass.renderpass.depthstencilAttachment
		if 0 <= depth_attach_idx < attachment_count:
			resource_ids.append(attachments[depth_attach_idx].imageResourceId)

		return resource_ids

	def save_output_targets(self, output_dir, clamp_pixel_range=False):
		current_pass = self.pipe_state.currentPass
		attachments = current_pass.framebuffer.attachments
		event_id = self.action.eventId

		for idx in current_pass.renderpass.colorAttachments:
			resource_id = attachments[idx].imageResourceId
			if resource_id == rd.ResourceId.Null():
				continue

			filepath = os.path.join(output_dir, f'draw_color_{event_id:04d}_{idx}')
			image_type = self._get_resource_image_type(resource_id, clamp_pixel_range)
			if _save_texture(resource_id, self.controller, filepath, image_type):
				print(f'└ Save output target {filepath}')

		depth_attach_idx = current_pass.renderpass.depthstencilAttachment
		if depth_attach_idx < 0:
			return

		resource_id = attachments[depth_attach_idx].imageResourceId
		image_type = self._get_resource_image_type(resource_id, clamp_pixel_range)
		filepath = os.path.join(output_dir, f'draw_depth_{event_id:04d}')
		if _save_texture(resource_id, self.controller, filepath, image_type):
			print(f'└ Save output target {filepath}')

	def get_viewport_info(self):
		viewport_states = []
		for viewport_scissor in self.pipe_state.viewportScissor.viewportScissors:
			viewport = viewport_scissor.vp
			if viewport.enabled:
				viewport_states.append(f'{int(viewport.width)} x {int(viewport.height)}')
		return '\n'.join(viewport_states)

	def get_renderpass_id(self):
		return int(self.pipe_state.currentPass.renderpass.resourceId)

	def get_input_texture_desc(self, shader_stage):
		input_tex_descs = []
		for resource_id in self._get_shader_read_tex_resource_ids(shader_stage):
			name, tex_desc = self._get_texture_info(resource_id)
			info = f'{name}: {tex_desc.width} x {tex_desc.height}, {tex_desc.format.Name()}'
			input_tex_descs.append(info)

		return '\n'.join(input_tex_descs)

	def get_output_target_desc(self):
		output_tex_descs = []
		# We can't directly retrieve the output targets from action (i.e. action.outputs)
		for resource_id in self._get_output_target_resource_ids():
			if resource_id == rd.ResourceId.Null():
				continue

			name, tex_desc = self._get_texture_info(resource_id)
			if tex_desc:
				info = f'{name}: {tex_desc.width} x {tex_desc.height}, {tex_desc.format.Name()}'
				output_tex_descs.append(info)

		return '\n'.join(output_tex_descs)

	def get_color_write_masks(self):
		results = []
		blends = self.pipe_state.colorBlend.blends
		for blend in blends:
			mask = blend.writeMask
			channels = ['R', 'G', 'B', 'A']
			writeMaskTokens = []
			for idx, ch in enumerate(channels):
				token = ch if mask & (1 << idx) else '_'
				writeMaskTokens.append(token)

			results.append(''.join(writeMaskTokens))

		return results

	def _append_draw_state_summary(self, action_name, action_elapsed_time, csv_writer):
		draw_state = DrawCallState(self.action, action_name, action_elapsed_time)
		draw_state.viewport_size = self.get_viewport_info()
		draw_state.renderpass_id = self.get_renderpass_id()

		if draw_state.renderpass_id != VKDrawStateExtractor.last_render_pass_id:
			draw_state.pass_switch = 'v'
		VKDrawStateExtractor.last_render_pass_id = draw_state.renderpass_id

		draw_state.vert_shader_id = int(_get_shader_resource_id(self.pipe_state.vertexShader))
		draw_state.frag_shader_id = int(_get_shader_resource_id(self.pipe_state.fragmentShader))
		draw_state.comp_shader_id = int(_get_shader_resource_id(self.pipe_state.computeShader))
		draw_state.vs_textures = self.get_input_texture_desc(rd.ShaderStage.Vertex)
		draw_state.fs_textures = self.get_input_texture_desc(rd.ShaderStage.Fragment)
		draw_state.output_targets = self.get_output_target_desc()

		write_masks = self.get_color_write_masks()
		draw_state.color_mask = '\n'.join(write_masks)

		depth_state = self.pipe_state.depthStencil
		draw_state.depth_state = _get_depth_function_desc(depth_state)
		draw_state.depth_write = 'v' if depth_state.depthWriteEnable else ''
		draw_state.write_to_csv_dict(csv_writer)


def traverse_draw_action(action: rd.ActionDescription, count: int, start_event_id: int, end_event_id: int):
	while action and count > 0:
		if action.eventId <= start_event_id:
			action = action.next
			continue
		elif end_event_id > 0 and action.eventId >= end_event_id:
			raise StopIteration

		if action.flags & (rd.ActionFlags.Drawcall | rd.ActionFlags.Dispatch):
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

	if not os.path.exists(options.output_dir):
		os.makedirs(options.output_dir)

	state = controller.GetPipelineState()
	if state.IsCaptureGL():
		_DRAW_STATE_LABEL_MAP['renderpass_id'] = 'FBO Id'

	# Construct map for (eventId, elapsed time) by using GPU timestamp.
	action_elapsed_time_map = {}
	if options.capture_gpu_duration:
		assert rd.GPUCounter.EventGPUDuration in controller.EnumerateCounters(), 'Not found GPU timestamp counters'

		for sample in controller.FetchCounters([rd.GPUCounter.EventGPUDuration]):
			action_elapsed_time_map[sample.eventId] = sample.value.d * 1000.0 		# Convert unit to millisecond

	visited_draws = 0
	with open(report_csv_path, 'w', newline='') as csvfile:
		csv_writer = csv.DictWriter(csvfile, fieldnames=_DRAW_STATE_LABEL_MAP.keys())
		csv_writer.writerow(_DRAW_STATE_LABEL_MAP)

		action = _get_first_action(controller)
		for action in traverse_draw_action(action, options.draw_count, options.start_event_id, options.end_event_id):
			if (visited_draws & 0x0F) == 0 or _VERBOSE:
				print(f'Extracting draw [EID: {action.eventId}]')
			visited_draws += 1

			controller.SetFrameEvent(action.eventId, True)
			if state.IsCaptureGL():
				draw_state_extractor = GLDrawStateExtractor(capture, controller, action)
			elif state.IsCaptureVK():
				draw_state_extractor = VKDrawStateExtractor(capture, controller, action)

			draw_state_extractor.extract_resources(options)

			action_elapsed_time = action_elapsed_time_map.get(action.eventId, None)
			draw_state_extractor.write_summary(action_elapsed_time, csv_writer)

	print(f'Finish exporting {capture_filename}')


def async_export(ctx: qrd.CaptureContext, options: ExportOptions):
	# Update spirv-cross path from config.
	global _SPIRV_CROSS_PATH
	for proc in ctx.Config().ShaderProcessors:
		if proc.tool == rd.KnownShaderTool.SPIRV_Cross:
			_SPIRV_CROSS_PATH = proc.executable
			break

	def _replay_callback(r: rd.ReplayController):
		export_draw_call_states(r, ctx, options)
	ctx.Replay().AsyncInvoke('DrawCallReporter', _replay_callback)


if 'pyrenderdoc' in globals():
	options = ExportOptions()
	options.draw_count = 30
	# options.start_event_id = 1885
	# options.end_event_id = 1990
	# options.export_input_textures = True
	# options.export_output_targets = True
	# options.export_shaders = True
	options.output_dir = r'D:\rdc_test\test'
	pyrenderdoc.Replay().BlockInvoke(lambda ctx: export_draw_call_states(ctx, pyrenderdoc, options))
