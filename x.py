import shutil

shutil.copytree('toolboxes/toolboxes', 'toolboxes', dirs_exist_ok=True)
shutil.rmtree('toolboxes/toolboxes')
shutil.copytree('core/core', 'core', dirs_exist_ok=True)
shutil.rmtree('core/core')
shutil.copytree('test/test', 'test', dirs_exist_ok=True)
shutil.rmtree('test/test')