/**
 * ================================================================================
 * STATIC SITE GENERATOR - build.js
 * ================================================================================
 * 
 * PURPOSE:
 * This script is the heart of the static site generator. It reads markdown files
 * from the `writing/` directory, converts them to HTML, applies templates from
 * `theme/templates/`, and writes the final output to `public/`.
 * 
 * ARCHITECTURE OVERVIEW:
 * 
 *   writing/          theme/templates/       public/
 *   ├── index.md  ──┐    ├── base.html       ├── index.html
 *   ├── about.md  ──┼────┤ index.html   ──>  ├── about.html
 *   └── posts/    ──┘    ├── post.html       └── posts/
 *       ├── *.md         └── about.html          └── *.html
 * 
 * WORKFLOW:
 * 1. Parse all markdown files
 * 2. Extract YAML frontmatter (title, description, etc.)
 * 3. Convert markdown body to HTML using 'marked' library
 * 4. Inject content into appropriate template
 * 5. Write final HTML to public/
 * 6. Copy static assets (CSS, JS)
 * 
 * RUNNING THIS SCRIPT:
 *   node scripts/build.js
 * 
 * DEPENDENCIES:
 * - Node.js (https://nodejs.org)
 * - marked: Markdown parser (installed via npm)
 * - gray-matter: YAML frontmatter parser (installed via npm)
 * 
 * FIRST-TIME SETUP:
 *   cd scripts
 *   npm install
 * 
 * ================================================================================
 */

// ================================================================================
// IMPORTS
// ================================================================================
// Node.js has a module system. We use `require()` to import functionality.
// - fs: File system operations (read/write files)
// - path: Cross-platform path manipulation
// - marked: Markdown to HTML conversion
// - matter: YAML frontmatter parsing
// ================================================================================

const fs = require('fs');
const path = require('path');

// 'marked' is a fast, low-level markdown parser
// We'll install it via npm in the scripts/ folder
const { marked } = require('marked');
const markedFootnote = require('marked-footnote');

// 'gray-matter' parses YAML frontmatter from markdown files
// Frontmatter is the metadata at the top of each .md file between --- delimiters
const matter = require('gray-matter');

// ================================================================================
// CONFIGURATION
// ================================================================================
// These constants define the directory structure. Having them in one place makes
// the script easier to modify if you want to reorganize later.
// 
// path.resolve() converts relative paths to absolute paths based on the project
// root, which we determine by going up one directory from scripts/
// ================================================================================

/** Project root directory (one level up from scripts/) */
const PROJECT_ROOT = path.resolve(__dirname, '..');

/** Root directory for markdown source files */
const WRITING_DIR = path.join(PROJECT_ROOT, 'writing');

/** Root directory for templates and static assets */
const THEME_DIR = path.join(PROJECT_ROOT, 'theme');

/** Output directory for generated HTML */
const PUBLIC_DIR = path.join(PROJECT_ROOT, 'public');

/** Subdirectory containing post markdown files */
const POSTS_SUBDIR = 'posts';

// ================================================================================
// MARKED CONFIGURATION
// ================================================================================
// We configure the 'marked' library to output HTML that works well with our
// styling. The key customizations:
// - Add language classes to code blocks for Prism.js syntax highlighting
// - Keep the output relatively clean
// ================================================================================

marked.setOptions({
    // Enable GitHub Flavored Markdown features (tables, strikethrough, etc.)
    gfm: true,

    // Don't automatically break on single newlines (require double newline for <p>)
    breaks: false,

    // Custom renderer for code blocks to add language class for Prism.js
    // This is how syntax highlighting knows what language to use
});

// Enable GFM footnotes
// Enable GFM (GitHub Flavored Markdown) footnotes
// This extension handles the [^1] syntax and automatically appends 
// a <section class="footnotes">...</section> to the end of the rendered HTML.
// This is why we don't need to manually inject footnotes in the template.
marked.use(markedFootnote());

/**
 * Custom code block renderer that adds Prism.js-compatible classes.
 * 
 * When you write:
 *   ```python
 *   def hello():
 *       print("world")
 *   ```
 * 
 * This renderer outputs:
 *   <pre><code class="language-python">...</code></pre>
 * 
 * The "language-python" class tells Prism.js to highlight as Python.
 */
const renderer = new marked.Renderer();
renderer.code = function (code, language) {
    const langClass = language ? ` class="language-${language}"` : '';
    // Escape HTML entities to prevent XSS and display issues
    const escapedCode = code
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    return `<pre><code${langClass}>${escapedCode}</code></pre>\n`;
};

marked.setOptions({ renderer });

// ================================================================================
// DATA STRUCTURES
// ================================================================================
// In JavaScript, we don't have case classes like Scala, but we can document
// the shape of our data. A "document" object has these properties:
// 
// {
//   title: string,       // Page title (browser tab, <h1>)
//   subtitle: string,    // Optional subtitle/tagline
//   description: string, // Meta description for SEO
//   date: string,        // Publication date
//   template: string,    // Which template: "post", "index", or "about"
//   usesMath: boolean,   // Include MathJax?
//   usesCode: boolean,   // Include Prism.js?
//   body: string,        // HTML-converted markdown body
//   slug: string         // URL-friendly filename
// }
// ================================================================================

// ================================================================================
// MARKDOWN PROCESSING
// ================================================================================

/**
 * Parse a markdown file and return a document object.
 * 
 * This function:
 * 1. Reads the raw file content
 * 2. Uses gray-matter to split frontmatter from body
 * 3. Converts the markdown body to HTML
 * 4. Returns a document object with all metadata
 * 
 * @param {string} filePath - Absolute path to the markdown file
 * @returns {Object} Document object with title, body, etc.
 */
function parseDocument(filePath) {
    // Read the raw file content
    const content = fs.readFileSync(filePath, 'utf-8');

    // gray-matter parses the frontmatter and returns:
    // - data: the parsed YAML as an object
    // - content: everything after the frontmatter
    const { data, content: markdown } = matter(content);

    // Convert markdown to HTML
    const body = marked.parse(markdown);

    // Generate slug from filename (remove .md extension)
    const slug = path.basename(filePath, '.md');

    // Return the document object with defaults for missing fields
    return {
        title: data.title || 'Untitled',
        subtitle: data.subtitle || '',
        description: data.description || '',
        date: data.date || '',
        template: data.template || 'post',
        usesMath: data.uses_math === true,
        usesCode: data.uses_code === true,
        body: body,
        slug: slug
    };
}

// ================================================================================
// TEMPLATE ENGINE
// ================================================================================
// Our templates use a simple {{placeholder}} syntax for variable substitution.
// This is intentionally primitive - no conditionals, no loops, just find-replace.
// 
// For a more complex site, you might use a real template engine like Handlebars
// or EJS. But simple string replacement is:
// - Easy to understand
// - Zero additional dependencies (we use it with base Node.js)
// - Sufficient for our needs
// ================================================================================

/**
 * Replace all {{placeholder}} occurrences in a template with actual values.
 * 
 * @param {string} template - The template string with {{placeholders}}
 * @param {Object} values - Object with key-value pairs for replacement
 * @returns {string} The template with all placeholders replaced
 */
function renderTemplate(template, values) {
    let result = template;
    for (const [key, value] of Object.entries(values)) {
        // Create a regex that matches {{key}} globally, allowing whitespace
        // The 'g' flag replaces ALL occurrences, not just the first
        const placeholder = new RegExp(`\\{\\{\\s*${key}\\s*\\}\\}`, 'g');
        // IMPORTANT: Use a function for the second argument to prevent
        // special interpretation of '$' characters (needed for MathJax config)
        result = result.replace(placeholder, () => value);
    }
    return result;
}

/**
 * Load a template file from the theme/templates directory.
 * 
 * @param {string} name - The template filename (e.g., "base.html")
 * @returns {string} The template content as a string
 */
function loadTemplate(name) {
    const templatePath = path.join(THEME_DIR, 'templates', name);
    return fs.readFileSync(templatePath, 'utf-8');
}

// ================================================================================
// HEAD EXTRAS - MathJax and Prism.js
// ================================================================================
// Posts that use math or code need extra scripts in the <head>.
// We generate these based on the frontmatter flags uses_math and uses_code.
// ================================================================================

/** MathJax configuration and script tags for LaTeX rendering */
const MATHJAX_SCRIPTS = `
  <!-- MathJax for LaTeX equations -->
  <script>
    MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      },
      svg: { fontCache: 'global' }
    };
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
`;

/** Prism.js script tags for code syntax highlighting */
const PRISMJS_SCRIPTS = `
  <!-- Prism.js for syntax highlighting -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
`;

/**
 * Generate the head_extra content based on document needs.
 * 
 * @param {Object} doc - Document object with usesMath and usesCode flags
 * @returns {string} HTML to inject in the <head>
 */
function generateHeadExtras(doc) {
    let extras = '';
    if (doc.usesMath) extras += MATHJAX_SCRIPTS;
    if (doc.usesCode) extras += PRISMJS_SCRIPTS;
    return extras;
}

// ================================================================================
// PAGE RENDERING
// ================================================================================
// These functions combine templates with content to produce final HTML pages.
// ================================================================================

/**
 * Render a post document to complete HTML.
 * 
 * This combines the post template with the base template to produce a full page.
 * 
 * @param {Object} doc - Document object from parseDocument()
 * @returns {string} Complete HTML page
 */
function renderPost(doc) {
    // Load both templates
    const baseTemplate = loadTemplate('base.html');
    const postTemplate = loadTemplate('post.html');

    // First, fill in the post template (inner content)
    const postContent = renderTemplate(postTemplate, {
        title: doc.title,
        subtitle: doc.subtitle,
        body: doc.body,
        // The 'footnotes' placeholder in post.html is deprecated because 
        // marked-footnote now appends footnotes directly to the 'body' HTML.
        // We pass an empty string to ensure the placeholder {{footnotes}} 
        // doesn't appear as raw text if it still exists in the template.
        footnotes: ''
    });

    // Then, fill in the base template (outer shell)
    return renderTemplate(baseTemplate, {
        title: doc.title,
        description: doc.description,
        head_extra: generateHeadExtras(doc),
        body_class: 'post-page',
        content: postContent
    });
}

/**
 * Format a post date for index display.
 * 
 * @param {string|Date} value - Date from frontmatter.
 * @returns {string} Formatted date like "Sat Dec 27 '25".
 */
function formatPostDate(value) {
    if (!value) return '';

    let date;
    if (value instanceof Date) {
        date = value;
    } else {
        const parsed = new Date(value);
        if (Number.isNaN(parsed.getTime())) {
            return String(value);
        }
        date = parsed;
    }

    if (Number.isNaN(date.getTime())) {
        return '';
    }

    const weekdays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const weekday = weekdays[date.getUTCDay()] || '';
    const month = months[date.getUTCMonth()] || '';
    const day = date.getUTCDate();
    const year = String(date.getUTCFullYear()).slice(-2);

    return `${weekday} ${month} ${day} '${year}`.trim();
}

/**
 * Render the index page with a list of all posts.
 * 
 * @param {Array} posts - Array of document objects
 * @returns {string} Complete HTML page
 */
function renderIndex(posts) {
    const baseTemplate = loadTemplate('base.html');
    const indexTemplate = loadTemplate('index.html');

    // Parse index.md for metadata if it exists
    const indexMdPath = path.join(WRITING_DIR, 'index.md');
    let indexDoc;
    if (fs.existsSync(indexMdPath)) {
        indexDoc = parseDocument(indexMdPath);
    } else {
        indexDoc = {
            title: 'Ink Dreams - Notes',
            description: 'Notes and essays in a hand-drawn aesthetic.'
        };
    }

    // Sort posts by date descending (newest first)
    // Sort posts by date descending (newest first)
    const sortedPosts = [...posts].sort((a, b) => {
        const dateA = new Date(a.date || 0);
        const dateB = new Date(b.date || 0);
        return dateB - dateA;
    });

    // Generate list items for each post
    // Links are now directory-based (no .html extension needed)
    const postListHtml = sortedPosts
        .map(post => {
            const formattedDate = formatPostDate(post.date);
            const dateHtml = formattedDate
                ? `<span class="post-date">${formattedDate}</span>`
                : '<span class="post-date"></span>';
            const titleHtml = `<a href="posts/${post.slug}/">${post.title}</a>`;
            return `  <li class="post-list-item">${dateHtml}${titleHtml}</li>`;
        })
        .join('\n');

    // Fill in templates
    const indexContent = renderTemplate(indexTemplate, {
        post_list: postListHtml
    });

    return renderTemplate(baseTemplate, {
        title: indexDoc.title,
        description: indexDoc.description,
        head_extra: '',
        body_class: 'centered-list',
        content: indexContent
    });
}

/**
 * Render the about page.
 * 
 * @returns {string} Complete HTML page
 */
function renderAbout() {
    const baseTemplate = loadTemplate('base.html');
    const aboutTemplate = loadTemplate('about.html');

    // Parse about.md for content
    const aboutMdPath = path.join(WRITING_DIR, 'about.md');
    let aboutDoc;
    if (fs.existsSync(aboutMdPath)) {
        aboutDoc = parseDocument(aboutMdPath);
    } else {
        aboutDoc = {
            title: 'About',
            description: 'About this site',
            body: '<p>About page content.</p>'
        };
    }

    const aboutContent = renderTemplate(aboutTemplate, {
        title: aboutDoc.title,
        content: aboutDoc.body
    });

    // On the about page, switch the footer to a subtle back arrow placed before the name.
    const footerReplacement =
        '<div class="fixed-footer">' +
        '<a href="/index.html" class="footer-back-arrow" aria-label="Back to notes"></a>' +
        '<span>&copy; Canaan McKenzie</span>' +
        '</div>';
    const baseWithModifiedFooter = baseTemplate.replace(
        /<div class="fixed-footer">[\s\S]*?<\/div>/,
        footerReplacement
    );

    return renderTemplate(baseWithModifiedFooter, {
        title: aboutDoc.title,
        description: aboutDoc.description,
        head_extra: '',
        body_class: 'hero',
        content: aboutContent
    });
}

/**
 * Render the 404 error page.
 * 
 * @returns {string} Complete HTML page
 */
function render404() {
    const baseTemplate = loadTemplate('base.html');
    const notFoundTemplate = loadTemplate('404.html');

    return renderTemplate(baseTemplate, {
        title: '404 Not Found',
        description: 'Page not found',
        head_extra: '',
        body_class: 'error-page',
        content: notFoundTemplate
    });
}

// ================================================================================
// FILE SYSTEM OPERATIONS
// ================================================================================
// Utility functions for working with the file system.
// ================================================================================

/**
 * Ensure a directory exists, creating it recursively if necessary.
 * 
 * @param {string} dirPath - Path to the directory
 */
function ensureDir(dirPath) {
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
    }
}

/**
 * Write content to a file and log it.
 * 
 * @param {string} filePath - Path to write to
 * @param {string} content - Content to write
 */
function writeFile(filePath, content) {
    fs.writeFileSync(filePath, content, 'utf-8');
    // Log relative path for cleaner output
    const relativePath = path.relative(PROJECT_ROOT, filePath);
    console.log(`  + Wrote: ${relativePath}`);
}

/**
 * Copy a file from source to destination.
 * 
 * @param {string} src - Source path
 * @param {string} dest - Destination path
 */
function copyFile(src, dest) {
    fs.copyFileSync(src, dest);
    const relSrc = path.relative(PROJECT_ROOT, src);
    const relDest = path.relative(PROJECT_ROOT, dest);
    console.log(`  + Copied: ${relSrc} -> ${relDest}`);
}

/**
 * Get all files with a given extension in a directory.
 * 
 * @param {string} dir - Directory to search
 * @param {string} ext - File extension (e.g., '.md')
 * @returns {Array} Array of absolute file paths
 */
function getFilesWithExtension(dir, ext) {
    if (!fs.existsSync(dir)) return [];

    return fs.readdirSync(dir)
        .filter(file => file.endsWith(ext))
        .map(file => path.join(dir, file));
}

// ================================================================================
// MAIN BUILD FUNCTION
// ================================================================================
// This is the entry point that orchestrates the entire build process.
// ================================================================================

/**
 * Build the entire site.
 * 
 * This function:
 * 1. Creates the public directory structure
 * 2. Parses all markdown posts
 * 3. Renders each post with its template
 * 4. Renders the index page with post list
 * 5. Renders the about page
 * 6. Copies static assets (CSS, JS)
 */
function buildSite() {
    console.log('='.repeat(60));
    console.log('BUILDING SITE');
    console.log('='.repeat(60));

    // Step 1: Prepare output directories
    console.log('\n[1/5] Preparing output directories...');
    ensureDir(PUBLIC_DIR);
    ensureDir(path.join(PUBLIC_DIR, POSTS_SUBDIR));

    // Step 2: Parse all posts
    console.log('\n[2/5] Parsing markdown posts...');
    const postsDir = path.join(WRITING_DIR, POSTS_SUBDIR);
    const postFiles = getFilesWithExtension(postsDir, '.md');
    console.log(`  Found ${postFiles.length} posts`);

    const posts = postFiles.map(filePath => {
        console.log(`  Parsing: ${path.basename(filePath)}`);
        return parseDocument(filePath);
    });

    // Step 3: Render posts
    console.log('\n[3/5] Rendering posts...');
    posts.forEach(doc => {
        const html = renderPost(doc);

        // Clean URLs: Create a directory for each post and write index.html inside
        const postDir = path.join(PUBLIC_DIR, POSTS_SUBDIR, doc.slug);
        ensureDir(postDir);

        const outputPath = path.join(postDir, 'index.html');
        writeFile(outputPath, html);
    });

    // Step 4: Render index, about, and 404 pages
    console.log('\n[4/5] Rendering main pages...');
    writeFile(path.join(PUBLIC_DIR, 'index.html'), renderIndex(posts));
    writeFile(path.join(PUBLIC_DIR, 'about.html'), renderAbout());
    writeFile(path.join(PUBLIC_DIR, '404.html'), render404());

    // Step 5: Copy static assets
    console.log('\n[5/5] Copying static assets...');
    copyFile(
        path.join(THEME_DIR, 'styles.css'),
        path.join(PUBLIC_DIR, 'styles.css')
    );
    copyFile(
        path.join(THEME_DIR, 'script.js'),
        path.join(PUBLIC_DIR, 'script.js')
    );
    copyFile(
        path.join(THEME_DIR, 'favicon.svg'),
        path.join(PUBLIC_DIR, 'favicon.svg')
    );

    console.log('\n' + '='.repeat(60));
    console.log('BUILD COMPLETE!');
    console.log('='.repeat(60));
    console.log(`\nGenerated ${posts.length} posts + index + about`);
    console.log(`Output directory: public/`);
    console.log('\nTo preview: node scripts/serve.js');
}

// ================================================================================
// ENTRY POINT
// ================================================================================
// When you run `node scripts/build.js`, this is what executes.
// ================================================================================

buildSite();
