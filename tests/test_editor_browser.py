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
        page.get_by_placeholder('dog barking, music').fill('dog')
        page.get_by_role('button', name='Separate audio').click()
        page.get_by_text('Ready — test stems; no AI separation was performed.').wait_for(timeout=30000)
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
