"""Opt-in Chromium test using real decoded video and exported YOLO metadata."""
import json
import os
from pathlib import Path
import pytest
from soundcanvas_cv.selection import select_at_point

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.browser, pytest.mark.skipif(os.getenv('SC_CV_BROWSER_TESTS') != '1', reason='Set SC_CV_BROWSER_TESTS=1 after generating real outputs and installing Chromium')]


def test_real_video_click_export_and_mismatch(tmp_path):
    from playwright.sync_api import sync_playwright
    metadata = json.loads((ROOT/'outputs/real_yolo/detections.json').read_text())
    obj = next(o for o in metadata['frames'][0]['objects'] if o['label'] == 'person')
    x1,y1,x2,y2 = obj['bbox_normalized']; x,y = (x1+x2)/2,(y1+y2)/2
    expected = select_at_point(metadata, 0, x, y)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width':1100,'height':1000})
        errors = []; page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto((ROOT/'viewer.html').as_uri())
        page.locator('#videoFile').set_input_files(ROOT/'examples/media/pedestrians.mp4')
        page.locator('#jsonFile').set_input_files(ROOT/'outputs/real_yolo/detections.json')
        page.wait_for_function("document.querySelector('#status').textContent.includes('Click a box')")
        page.wait_for_function("document.querySelector('video').readyState>=2 && !document.querySelector('video').seeking")
        pos = page.evaluate('''([x,y]) => { const v=document.querySelector('video'),r=v.getBoundingClientRect();
          const c=SoundCanvasCV.contentRect(r.width,r.height,v.videoWidth,v.videoHeight);
          return {x:r.left+c.x+x*c.w,y:r.top+c.y+y*c.h}; }''', [x,y])
        page.mouse.click(pos['x'],pos['y'])
        page.wait_for_function("!document.querySelector('#export').disabled")
        actual = json.loads(page.locator('#selectionJson').text_content())
        assert actual == expected
        page.evaluate("document.querySelector('video').currentTime=3")
        page.wait_for_function("document.querySelector('video').currentTime===3 && !document.querySelector('video').seeking && document.querySelector('video').readyState>=2")
        with page.expect_download() as event: page.locator('#export').click()
        path = tmp_path/'selection.json'; event.value.save_as(path)
        assert json.loads(path.read_text()) == expected  # export remains frozen after seek
        page.screenshot(path=str(ROOT/'outputs/real_yolo/viewer.png'), full_page=True)
        assert page.evaluate('SoundCanvasCV.frameAtTime('+json.dumps(metadata)+',12)') is None
        # Invalid JSON clears the prior selection and metadata.
        bad = tmp_path/'bad.json'; bad.write_text('{}')
        page.locator('#jsonFile').set_input_files(bad)
        page.wait_for_function("document.querySelector('#status').textContent.startsWith('Invalid JSON')")
        assert page.locator('#export').is_disabled()
        mismatch = {**metadata, 'source_sha256':'0'*64}
        bad = tmp_path/'mismatch.json'; bad.write_text(json.dumps(mismatch)); page.locator('#jsonFile').set_input_files(bad)
        page.wait_for_function("document.querySelector('#status').textContent.includes('does not match')")
        assert not errors, errors
        browser.close()
