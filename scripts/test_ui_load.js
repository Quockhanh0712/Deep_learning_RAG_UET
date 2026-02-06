/**
 * Smoke Test: Verify Frontend Dev Server
 * 
 * Usage: node scripts/test_ui_load.js
 * 
 * Checks:
 * 1. Server responds with HTTP 200
 * 2. Response contains expected HTML elements
 */

const http = require('http');

const URL = 'http://localhost:5173/';
const TIMEOUT = 5000;

function testUILoad() {
    return new Promise((resolve, reject) => {
        const req = http.get(URL, { timeout: TIMEOUT }, (res) => {
            let body = '';

            res.on('data', chunk => body += chunk);
            res.on('end', () => {
                const checks = {
                    httpStatus: res.statusCode === 200,
                    hasHtml: body.includes('<!DOCTYPE html>') || body.includes('<html'),
                    hasRoot: body.includes('id="root"'),
                    hasScript: body.includes('<script'),
                };

                const allPassed = Object.values(checks).every(Boolean);

                console.log('\n🧪 Frontend Smoke Test Results\n');
                console.log(`URL: ${URL}`);
                console.log(`HTTP Status: ${res.statusCode} ${checks.httpStatus ? '✅' : '❌'}`);
                console.log(`Has HTML Structure: ${checks.hasHtml ? '✅' : '❌'}`);
                console.log(`Has Root Element: ${checks.hasRoot ? '✅' : '❌'}`);
                console.log(`Has Script Tag: ${checks.hasScript ? '✅' : '❌'}`);
                console.log(`\n${allPassed ? '✅ ALL CHECKS PASSED' : '❌ SOME CHECKS FAILED'}\n`);

                resolve(allPassed);
            });
        });

        req.on('error', (err) => {
            console.error(`\n❌ Connection Error: ${err.message}`);
            console.error('Make sure the dev server is running: npm run dev\n');
            reject(err);
        });

        req.on('timeout', () => {
            req.destroy();
            reject(new Error('Request timed out'));
        });
    });
}

testUILoad()
    .then(passed => process.exit(passed ? 0 : 1))
    .catch(() => process.exit(1));
