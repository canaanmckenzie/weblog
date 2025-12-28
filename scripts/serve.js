/**
 * ================================================================================
 * DEVELOPMENT SERVER - serve.js
 * ================================================================================
 * 
 * PURPOSE:
 * This script starts a local HTTP server to preview the built site.
 * It serves static files from the `public/` directory.
 * 
 * RUNNING THIS SCRIPT:
 *   node scripts/serve.js
 * 
 * Then open http://localhost:8000 in your browser.
 * 
 * WHY NOT JUST OPEN THE HTML FILES DIRECTLY?
 * Opening HTML files with file:// protocol has limitations:
 * - Some JavaScript features don't work
 * - Relative paths may behave differently
 * - Browser security restrictions on local files
 * 
 * A proper HTTP server makes the preview behave like a real website.
 * 
 * DEPENDENCIES:
 * - Node.js (built-in http module, no external dependencies needed)
 * 
 * ================================================================================
 */

// ================================================================================
// IMPORTS
// ================================================================================
// We use only Node.js built-in modules, no npm packages needed for serving.
// - http: Create HTTP server
// - fs: Read files from disk
// - path: Handle file paths
// - url: Parse request URLs
// ================================================================================

const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

// ================================================================================
// CONFIGURATION
// ================================================================================

/** Port to run the server on */
const PORT = 8000;

/** Directory to serve files from */
const PUBLIC_DIR = path.resolve(__dirname, '..', 'public');

// ================================================================================
// MIME TYPE MAPPING
// ================================================================================
// When serving files, we need to tell the browser what type of content it's
// receiving. This is done via the Content-Type header.
// 
// MIME types are standardized identifiers for file formats:
// - text/html: HTML documents
// - text/css: Stylesheets
// - application/javascript: JavaScript files
// - etc.
// ================================================================================

const MIME_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
    '.ttf': 'font/ttf',
};

/**
 * Get the MIME type for a file based on its extension.
 * 
 * @param {string} filePath - Path to the file
 * @returns {string} MIME type string
 */
function getMimeType(filePath) {
    const ext = path.extname(filePath).toLowerCase();
    return MIME_TYPES[ext] || 'application/octet-stream';
}

// ================================================================================
// REQUEST HANDLER
// ================================================================================
// This function is called for every HTTP request the server receives.
// It:
// 1. Parses the requested URL
// 2. Maps it to a file in the public/ directory
// 3. Reads and serves the file (or returns 404 if not found)
// ================================================================================

/**
 * Handle an incoming HTTP request.
 * 
 * @param {http.IncomingMessage} req - The request object
 * @param {http.ServerResponse} res - The response object
 */
function handleRequest(req, res) {
    // Parse the URL to get the pathname
    const parsedUrl = url.parse(req.url);
    let pathname = parsedUrl.pathname;

    // Default to index.html for root path
    if (pathname === '/') {
        pathname = '/index.html';
    }

    // Security: Prevent directory traversal attacks
    // someone could try to request "/../../../etc/passwd"
    const safePath = path.normalize(pathname).replace(/^(\.\.[\/\\])+/, '');

    // Build the full file path
    const filePath = path.join(PUBLIC_DIR, safePath);

    // Check if the file exists and is within PUBLIC_DIR
    if (!filePath.startsWith(PUBLIC_DIR)) {
        // Attempted directory traversal
        res.writeHead(403, { 'Content-Type': 'text/plain' });
        res.end('Forbidden');
        return;
    }

    // Attempt to read the file
    // Strategy:
    // 1. Try reading standard file path
    // 2. If it's a directory (or missing extension), try adding /index.html
    // 3. If missing, serve 404.html

    fs.stat(filePath, (err, stats) => {
        let finalPath = filePath;

        // If it's a directory (e.g. /posts/slug/), assume index.html
        if (!err && stats.isDirectory()) {
            finalPath = path.join(filePath, 'index.html');
        }
        // If it doesn't exist, it might be a clean URL request (e.g. /posts/slug) 
        // that needs to look into a directory
        else if (err && err.code === 'ENOENT' && !path.extname(filePath)) {
            // check if filePath + /index.html exists
            const potentialIdx = path.join(filePath, 'index.html');
            if (fs.existsSync(potentialIdx)) {
                finalPath = potentialIdx;
            }
        }

        fs.readFile(finalPath, (readErr, data) => {
            if (readErr) {
                if (readErr.code === 'ENOENT') {
                    // File definitely not found. Serve custom 404.
                    console.log(`  404: ${pathname} -> serving 404.html`);
                    const notFoundPath = path.join(PUBLIC_DIR, '404.html');
                    fs.readFile(notFoundPath, (err404, data404) => {
                        res.writeHead(404, { 'Content-Type': 'text/html; charset=utf-8' });
                        if (!err404) {
                            res.end(data404);
                        } else {
                            res.end('<h1>404 Not Found</h1>');
                        }
                    });
                } else {
                    // Other error
                    console.error(`  Error reading ${pathname}:`, readErr.message);
                    res.writeHead(500, { 'Content-Type': 'text/plain' });
                    res.end('Internal Server Error');
                }
                return;
            }

            // Success! Serve the file
            console.log(`  200: ${pathname}`);
            res.writeHead(200, { 'Content-Type': getMimeType(finalPath) });
            res.end(data);
        });
    });
}


// ================================================================================
// SERVER STARTUP
// ================================================================================

/**
 * Start the HTTP server.
 */
function startServer() {
    // Check if public directory exists
    if (!fs.existsSync(PUBLIC_DIR)) {
        console.error('Error: public/ directory does not exist.');
        console.error('Run "node scripts/build.js" first to generate the site.');
        process.exit(1);
    }

    // Create and start the server
    const server = http.createServer(handleRequest);

    server.listen(PORT, () => {
        console.log('='.repeat(50));
        console.log('DEVELOPMENT SERVER');
        console.log('='.repeat(50));
        console.log(`\nServing: ${PUBLIC_DIR}`);
        console.log(`\nOpen in browser: http://localhost:${PORT}`);
        console.log('\nPress Ctrl+C to stop.');
        console.log('\n' + '-'.repeat(50));
        console.log('Request log:');
    });

    // Handle server errors
    server.on('error', (err) => {
        if (err.code === 'EADDRINUSE') {
            console.error(`Error: Port ${PORT} is already in use.`);
            console.error('Try stopping other servers or use a different port.');
        } else {
            console.error('Server error:', err.message);
        }
        process.exit(1);
    });
}

// ================================================================================
// ENTRY POINT
// ================================================================================

startServer();
