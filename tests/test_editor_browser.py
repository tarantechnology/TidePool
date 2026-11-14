"""Opt-in browser smoke test against a running fake-inference backend and editor."""
import json
import os
import subprocess

import pytest


def test_editor_browser_workflow(tmp_path):
    url = os.environ.get('SOUNDCANVAS_BROWSER_URL')
    if not url:
        pytest.skip('Set SOUNDCANVAS_BROWSER_URL to run the browser smoke test')
    browser_api = pytest.importorskip('playwright.sync_api')
    video = tmp_path / 'input.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=blue:s=128x128:r=10:d=2', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2', '-c:v', 'mpeg4', '-c:a', 'aac', '-shortest', str(video)], check=True)
    with browser_api.sync_playwright() as playwright:
        options = {'headless': True, 'args': ['--autoplay-policy=no-user-gesture-required']}
        if os.environ.get('SOUNDCANVAS_CHROMIUM'):
            options['executable_path'] = os.environ['SOUNDCANVAS_CHROMIUM']
        browser = playwright.chromium.launch(**options)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000}, accept_downloads=True)
        page.goto(url)
        page.get_by_label('Media', exact=True).set_input_files(str(video))
        page.get_by_role('button', name='Create & upload').click()
        page.get_by_text('Upload complete. Add a sound prompt and process.').wait_for(timeout=30000)
        page.get_by_placeholder('dog barking').fill('dog')
        page.get_by_role('button',name='I hear it here',exact=True).click()
        page.get_by_label('Sound present start',exact=True).fill('0.1')
        page.get_by_label('Sound present end',exact=True).fill('0.5')
        page.get_by_label('Edit after separation',exact=True).select_option('mute')
        page.get_by_label('Mute only an edit interval').check()
        page.get_by_label('Edit start',exact=True).fill('0.7')
        page.get_by_label('Edit end',exact=True).fill('1.2')
        page.get_by_role('button', name='Preview',exact=True).click()
        page.get_by_role('button',name='Apply preview',exact=True).wait_for(timeout=120000)
        page.get_by_role('button',name='Play preview',exact=True).click()
        page.get_by_role('button',name='Original mix',exact=True).click()
        page.get_by_role('button',name='Edited mix',exact=True).click()
        page.get_by_role('button',name='Pause preview',exact=True).click()
        page.get_by_role('button',name='Apply preview',exact=True).click()
        page.get_by_text('Preview applied. Existing layer edits were preserved.',exact=True).wait_for(timeout=30000)
        page.get_by_text('Try another separation',exact=True).click()
        page.get_by_role('button',name='Choose sound',exact=True).click()
        page.get_by_label('Retry sound description',exact=True).fill('dog barking')
        page.get_by_role('button',name='Quality replacement' if os.getenv('SOUNDCANVAS_BROWSER_QUALITY') else 'Preview replacement',exact=True).click()
        page.get_by_role('button',name='Apply preview',exact=True).wait_for(timeout=120000)
        page.get_by_role('button',name='Apply preview',exact=True).click()
        page.get_by_text('Preview applied. Existing layer edits were preserved.',exact=True).wait_for(timeout=30000)
        page.get_by_role('button',name='Undo applied preview',exact=True).click()
        page.get_by_text('Previous audio restored.',exact=True).wait_for(timeout=30000)
        page.locator('input[title="Volume (double-click to reset)"]').first.fill('0.5')
        page.get_by_text('Remote command test').click()
        page.get_by_placeholder('Mute the dog').fill('Mute the dog from 0.1 to 0.5 seconds')
        page.get_by_role('button', name='Send', exact=True).click()
        page.get_by_text('Edit applied to dog.').wait_for(timeout=30000)
        page.get_by_text('Stem settings', exact=True).click()
        page.get_by_role('button', name='Load stem settings').click()
        page.get_by_label('Clear remote mute ranges').check()
        page.get_by_role('button', name='Save stem settings').click()
        page.get_by_text('Stem settings saved. Reload settings to edit again.').wait_for(timeout=30000)
        with page.expect_download(timeout=30000) as download:
            page.get_by_role('button', name='Export', exact=True).click()
        output = tmp_path / 'output.mp4'
        download.value.save_as(str(output))
        streams = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type', '-of', 'json', str(output)]))['streams']
        assert {stream['codec_type'] for stream in streams} >= {'audio', 'video'}
        browser.close()


def test_visual_editor_with_real_gpu(tmp_path):
    """Opt-in: offset test video with a visible person and a running real GPU service."""
    url = os.environ.get('SOUNDCANVAS_BROWSER_URL')
    video = os.environ.get('SOUNDCANVAS_VISUAL_TEST_VIDEO')
    if not url or not video:
        pytest.skip('Set SOUNDCANVAS_BROWSER_URL and SOUNDCANVAS_VISUAL_TEST_VIDEO')
    browser_api = pytest.importorskip('playwright.sync_api')
    with browser_api.sync_playwright() as playwright:
        options = {'headless': True, 'args': ['--autoplay-policy=no-user-gesture-required']}
        if os.environ.get('SOUNDCANVAS_CHROMIUM'):
            options['executable_path'] = os.environ['SOUNDCANVAS_CHROMIUM']
        browser=playwright.chromium.launch(**options)
        page=browser.new_page(viewport={'width':1440,'height':1000},accept_downloads=True)
        page.goto(url)
        page.get_by_text('Mock stems',exact=True).wait_for(timeout=30000)
        page.locator('input[accept="video/*"]').set_input_files(video)
        page.get_by_text('Upload complete. Add a sound prompt and process.').wait_for(timeout=120000)
        page.get_by_role('button',name='Find objects in video').click()
        page.get_by_role('alert').get_by_text('Visual prompting requires audio/video starting at zero',exact=False).wait_for(timeout=30000)
        page.get_by_text('Visual selection options').click()
        page.get_by_role('button',name='Prepare compatible copy').click()
        page.get_by_text('Analysis ready for the first',exact=False).wait_for(timeout=180000)
        page.locator('button[aria-label="Select person"]').first.click()
        page.get_by_label('Selected object sound').fill('person speaking')
        page.get_by_label('Selected object actions').get_by_role('button',name='Mute').click()
        page.get_by_role('button',name='Apply preview',exact=True).wait_for(timeout=180000)
        page.get_by_role('button',name='Apply preview',exact=True).click()
        page.get_by_text('Preview applied.',exact=False).wait_for(timeout=30000)
        page.get_by_role('button',name='Compare original').click()
        assert page.get_by_role('button',name='Listening: original').count()==1
        page.get_by_role('button',name='Listening: original').click()
        page.get_by_label('Sound description',exact=True).fill('music')
        page.get_by_role('button',name='Preview',exact=True).click()
        page.get_by_role('button',name='Apply preview',exact=True).wait_for(timeout=180000)
        page.get_by_role('button',name='Apply preview',exact=True).click()
        page.get_by_text('Preview applied.',exact=False).wait_for(timeout=30000)
        with page.expect_download(timeout=120000) as download:
            page.get_by_role('button',name='Export',exact=True).click()
        output=tmp_path/'visual-export.mp4';download.value.save_as(str(output))
        streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_type','-of','json',str(output)]))['streams']
        assert {stream['codec_type'] for stream in streams} >= {'audio','video'}
        browser.close()
