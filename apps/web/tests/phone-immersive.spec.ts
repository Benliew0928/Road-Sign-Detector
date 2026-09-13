import {test,expect} from '@playwright/test';
test('immersive live camera fills phone and iPad viewports',async({page})=>{
 await page.addInitScript(()=>{navigator.mediaDevices.getUserMedia=()=>{const canvas=document.createElement('canvas');canvas.width=1280;canvas.height=720;const ctx=canvas.getContext('2d')!;ctx.fillStyle='#344b32';ctx.fillRect(0,0,1280,720);ctx.fillStyle='#869b74';ctx.fillRect(0,420,1280,300);return Promise.resolve(canvas.captureStream(10));};});
 await page.routeWebSocket('**/api/v1/ws/camera/**',()=>{});
 await page.goto('/phone?session=preview');await page.getByRole('button',{name:'Start stream',exact:true}).click();
 await expect(page.getByText('Streaming',{exact:true})).toBeVisible();
 for(const [width,height] of [[390,844],[820,1180],[1180,820],[844,390]]){
 await page.setViewportSize({width,height});
 await expect(page.locator('.phone-live-stage')).toHaveCSS('width',`${width}px`);
 await expect(page.locator('.phone-live-stage')).toHaveCSS('height',`${height}px`);
 await expect(page.getByRole('button',{name:'Stop stream',exact:true})).toBeVisible();
 expect(await page.evaluate(()=>document.documentElement.scrollHeight<=innerHeight)).toBe(true);
 await page.screenshot({path:`../../outputs/phone-immersive/live-${width}.png`,animations:'disabled'});
 }
 await page.getByRole('button',{name:'Stop stream',exact:true}).click();await expect(page.getByRole('button',{name:'Start stream',exact:true})).toBeVisible();
});
