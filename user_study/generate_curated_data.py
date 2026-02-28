"""
Generates curated_study_data.jsonl from hand-crafted code examples.

Activations are random placeholders — replace by running real SAE inference
and merging the results into the output file.

Run:
    python generate_curated_data.py
"""

import json
import random
import re
import sys
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).parent.parent / "sae_java_bug"
HYPOTHESES_FILE = ROOT / "sparse_autoencoders" / "my_hypotheses_layer11.jsonl"
OUT_DIR = Path(__file__).parent / "data"
OUT_FILE = OUT_DIR / "curated_study_data.jsonl"

TOP_K = 15          # features shown per example
N_TOTAL = 16384     # total SAE features

# ── Curated examples ───────────────────────────────────────────────────────────

EXAMPLES = [
    # ── CWE-89: SQL Injection ──────────────────────────────────────────────────
    {
        "id": "cwe89-java-001",
        "cwe": "CWE-89",
        "title": "SQL query via string concatenation",
        "file_extension": "java",
        "vulnerable_code": """\
public User getUser(Connection connection, String username) throws SQLException {
    // VULNERABLE: user input concatenated directly into the query
    String query = "SELECT * FROM users WHERE username = '" + username + "'";
    Statement stmt = connection.createStatement();
    ResultSet rs = stmt.executeQuery(query);
    if (rs.next()) {
        return new User(rs.getInt("id"), rs.getString("username"), rs.getString("email"));
    }
    return null;
}""",
        "secure_code": """\
public User getUser(Connection connection, String username) throws SQLException {
    // SECURE: parameterised query prevents SQL injection
    String query = "SELECT * FROM users WHERE username = ?";
    PreparedStatement stmt = connection.prepareStatement(query);
    stmt.setString(1, username);
    ResultSet rs = stmt.executeQuery();
    if (rs.next()) {
        return new User(rs.getInt("id"), rs.getString("username"), rs.getString("email"));
    }
    return null;
}""",
    },
    {
        "id": "cwe89-java-002",
        "cwe": "CWE-89",
        "title": "Dynamic ORDER BY clause injection",
        "file_extension": "java",
        "vulnerable_code": """\
public List<Product> getProducts(Connection connection, String sortColumn) throws SQLException {
    // VULNERABLE: sort column is user-controlled — cannot use a placeholder here,
    // but whitelist validation is still required
    String query = "SELECT id, name, price FROM products ORDER BY " + sortColumn;
    Statement stmt = connection.createStatement();
    ResultSet rs = stmt.executeQuery(query);
    List<Product> products = new ArrayList<>();
    while (rs.next()) {
        products.add(new Product(rs.getInt("id"), rs.getString("name"), rs.getDouble("price")));
    }
    return products;
}""",
        "secure_code": """\
private static final Set<String> ALLOWED_COLUMNS =
        Set.of("id", "name", "price", "created_at");

public List<Product> getProducts(Connection connection, String sortColumn) throws SQLException {
    // SECURE: column name validated against a whitelist before use
    if (!ALLOWED_COLUMNS.contains(sortColumn)) {
        throw new IllegalArgumentException("Invalid sort column: " + sortColumn);
    }
    String query = "SELECT id, name, price FROM products ORDER BY " + sortColumn;
    Statement stmt = connection.createStatement();
    ResultSet rs = stmt.executeQuery(query);
    List<Product> products = new ArrayList<>();
    while (rs.next()) {
        products.add(new Product(rs.getInt("id"), rs.getString("name"), rs.getDouble("price")));
    }
    return products;
}""",
    },

    # ── CWE-79: Cross-Site Scripting ───────────────────────────────────────────
    {
        "id": "cwe79-java-001",
        "cwe": "CWE-79",
        "title": "Reflected XSS via unescaped request parameter",
        "file_extension": "java",
        "vulnerable_code": """\
@WebServlet("/search")
public class SearchServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        // VULNERABLE: query parameter written directly to HTML output
        String query = request.getParameter("q");
        response.setContentType("text/html");
        PrintWriter out = response.getWriter();
        out.println("<html><body>");
        out.println("<h2>Search results for: " + query + "</h2>");
        out.println("</body></html>");
    }
}""",
        "secure_code": """\
@WebServlet("/search")
public class SearchServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        // SECURE: HTML-escape the parameter before writing to output
        String query = request.getParameter("q");
        String safeQuery = HtmlUtils.htmlEscape(query != null ? query : "");
        response.setContentType("text/html");
        PrintWriter out = response.getWriter();
        out.println("<html><body>");
        out.println("<h2>Search results for: " + safeQuery + "</h2>");
        out.println("</body></html>");
    }
}""",
    },
    {
        "id": "cwe79-java-002",
        "cwe": "CWE-79",
        "title": "Stored XSS — user comment rendered without escaping",
        "file_extension": "java",
        "vulnerable_code": """\
public String renderComments(List<Comment> comments) {
    StringBuilder html = new StringBuilder("<ul>");
    for (Comment comment : comments) {
        // VULNERABLE: stored comment content is written without escaping
        html.append("<li><strong>")
            .append(comment.getAuthor())
            .append(":</strong> ")
            .append(comment.getContent())
            .append("</li>");
    }
    html.append("</ul>");
    return html.toString();
}""",
        "secure_code": """\
public String renderComments(List<Comment> comments) {
    StringBuilder html = new StringBuilder("<ul>");
    for (Comment comment : comments) {
        // SECURE: both author and content are HTML-escaped before insertion
        html.append("<li><strong>")
            .append(HtmlUtils.htmlEscape(comment.getAuthor()))
            .append(":</strong> ")
            .append(HtmlUtils.htmlEscape(comment.getContent()))
            .append("</li>");
    }
    html.append("</ul>");
    return html.toString();
}""",
    },

    # ── CWE-20: Improper Input Validation ─────────────────────────────────────
    {
        "id": "cwe20-java-001",
        "cwe": "CWE-20",
        "title": "Array access without bounds or type check",
        "file_extension": "java",
        "vulnerable_code": """\
public String getItem(String[] items, String indexStr) {
    // VULNERABLE: no format check, no bounds check
    int index = Integer.parseInt(indexStr);
    return items[index];
}""",
        "secure_code": """\
public String getItem(String[] items, String indexStr) {
    // SECURE: validate format and bounds before accessing the array
    if (indexStr == null || indexStr.isBlank()) {
        throw new IllegalArgumentException("Index must not be empty");
    }
    int index;
    try {
        index = Integer.parseInt(indexStr.trim());
    } catch (NumberFormatException e) {
        throw new IllegalArgumentException("Index is not a valid integer: " + indexStr);
    }
    if (items == null || index < 0 || index >= items.length) {
        throw new IndexOutOfBoundsException("Index out of range: " + index);
    }
    return items[index];
}""",
    },
    {
        "id": "cwe20-java-002",
        "cwe": "CWE-20",
        "title": "File path traversal via unvalidated user input",
        "file_extension": "java",
        "vulnerable_code": """\
public byte[] readFile(String filename) throws IOException {
    // VULNERABLE: filename is used directly, allowing path traversal (e.g. ../../etc/passwd)
    File file = new File("/var/app/uploads/" + filename);
    return Files.readAllBytes(file.toPath());
}""",
        "secure_code": """\
private static final Path UPLOAD_DIR = Path.of("/var/app/uploads/").toAbsolutePath().normalize();

public byte[] readFile(String filename) throws IOException {
    // SECURE: canonicalise and confirm the resolved path stays inside the upload directory
    if (filename == null || filename.isBlank()) {
        throw new IllegalArgumentException("Filename must not be empty");
    }
    Path resolved = UPLOAD_DIR.resolve(filename).normalize();
    if (!resolved.startsWith(UPLOAD_DIR)) {
        throw new SecurityException("Access denied: path traversal detected");
    }
    return Files.readAllBytes(resolved);
}""",
    },

    # ── CWE-200: Information Exposure ─────────────────────────────────────────
    {
        "id": "cwe200-java-001",
        "cwe": "CWE-200",
        "title": "Stack trace leaked to HTTP response",
        "file_extension": "java",
        "vulnerable_code": """\
@PostMapping("/login")
public ResponseEntity<String> login(@RequestBody LoginRequest req) {
    try {
        User user = userService.authenticate(req.getUsername(), req.getPassword());
        String token = jwtService.generateToken(user);
        return ResponseEntity.ok(token);
    } catch (Exception e) {
        // VULNERABLE: full exception (including stack trace) sent to the client
        return ResponseEntity
                .status(HttpStatus.UNAUTHORIZED)
                .body("Login failed: " + e.getMessage() + "\\n" + Arrays.toString(e.getStackTrace()));
    }
}""",
        "secure_code": """\
@PostMapping("/login")
public ResponseEntity<String> login(@RequestBody LoginRequest req) {
    try {
        User user = userService.authenticate(req.getUsername(), req.getPassword());
        String token = jwtService.generateToken(user);
        return ResponseEntity.ok(token);
    } catch (Exception e) {
        // SECURE: log the detail server-side; return only a generic message to the client
        log.error("Authentication failed for user '{}': {}", req.getUsername(), e.getMessage(), e);
        return ResponseEntity
                .status(HttpStatus.UNAUTHORIZED)
                .body("Invalid credentials. Please try again.");
    }
}""",
    },
    {
        "id": "cwe200-java-002",
        "cwe": "CWE-200",
        "title": "Internal directory listing exposed via exception message",
        "file_extension": "java",
        "vulnerable_code": """\
public ResponseEntity<byte[]> downloadReport(String reportId) {
    try {
        File report = new File("/internal/reports/" + reportId + ".pdf");
        byte[] content = Files.readAllBytes(report.toPath());
        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_PDF)
                .body(content);
    } catch (IOException e) {
        // VULNERABLE: exception message may expose server-side paths
        return ResponseEntity.status(500).body(e.getMessage().getBytes());
    }
}""",
        "secure_code": """\
private static final Path REPORTS_DIR = Path.of("/internal/reports/").toAbsolutePath().normalize();

public ResponseEntity<byte[]> downloadReport(String reportId) {
    try {
        Path reportPath = REPORTS_DIR.resolve(reportId + ".pdf").normalize();
        if (!reportPath.startsWith(REPORTS_DIR)) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).build();
        }
        byte[] content = Files.readAllBytes(reportPath);
        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_PDF)
                .body(content);
    } catch (IOException e) {
        // SECURE: log the real error, return a generic message to the client
        log.error("Failed to read report '{}': {}", reportId, e.getMessage(), e);
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body("Report unavailable.".getBytes());
    }
}""",
    },
]

# ── Keywords used to bias feature selection toward each CWE ───────────────────

CWE_KEYWORDS = {
    "CWE-89":  ["sql", "injection", "query", "database", "concatenat"],
    "CWE-79":  ["xss", "html", "output", "render", "web", "script"],
    "CWE-20":  ["validation", "input", "sanitiz", "bounds", "check"],
    "CWE-200": ["information", "exposure", "error", "exception", "leak", "disclose"],
}


def load_hypotheses(path: Path) -> list[dict]:
    hypotheses = []
    print(f"Loading hypotheses from {path} …")
    with open(path) as f:
        for line in f:
            h = json.loads(line)
            hypotheses.append(h)
    print(f"  Loaded {len(hypotheses):,} features.")
    return hypotheses


def score_feature(feature: dict, keywords: list[str]) -> float:
    text = (
        (feature.get("hypothesis") or "")
        + " "
        + (feature.get("notes") or "")
    ).lower()
    return sum(text.count(kw) for kw in keywords)


def select_features(hypotheses: list[dict], cwe: str, k: int) -> list[dict]:
    """
    Pick k features: roughly half biased toward CWE-relevant hypotheses,
    the rest sampled from the full distribution.
    """
    keywords = CWE_KEYWORDS.get(cwe, [])
    scored = sorted(hypotheses, key=lambda h: score_feature(h, keywords), reverse=True)

    # top-k by keyword relevance (with some noise so we don't always pick the same ones)
    top_pool = scored[: k * 5]
    relevant = random.sample(top_pool, min(k // 2 + 1, len(top_pool)))

    # fill remainder from random features not already chosen
    chosen_ids = {h["feature_idx"] for h in relevant}
    rest_pool = [h for h in hypotheses if h["feature_idx"] not in chosen_ids]
    filler = random.sample(rest_pool, k - len(relevant))

    selected = relevant + filler
    random.shuffle(selected)
    return selected[:k]


def make_activations(features: list[dict], bias: str) -> list[dict]:
    """
    Generate plausible placeholder activations.
    'bias' is either 'vulnerable' or 'secure'.
    Features are sorted by |diff| descending before returning.
    """
    out = []
    for feat in features:
        base = random.uniform(0.0, feat.get("max_activation", 0.5) * 0.4)
        delta = random.uniform(0.05, feat.get("max_activation", 0.5) * 0.8)
        if bias == "vulnerable":
            sec_act = round(base, 6)
            vul_act = round(min(base + delta, feat.get("max_activation", 1.0)), 6)
        else:
            vul_act = round(base, 6)
            sec_act = round(min(base + delta, feat.get("max_activation", 1.0)), 6)
        diff = round(vul_act - sec_act, 6)
        out.append(
            {
                "feature_idx": feat["feature_idx"],
                "secure_activation": sec_act,
                "vulnerable_activation": vul_act,
                "diff": diff,
                "hypothesis": feat.get("hypothesis") or "No hypothesis available.",
                "confidence": feat.get("confidence") or "",
                "notes": feat.get("notes") or "",
                "n_nonzero": feat.get("n_nonzero", 0),
                "max_activation": feat.get("max_activation", 0.0),
            }
        )
    out.sort(key=lambda x: abs(x["diff"]), reverse=True)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not HYPOTHESES_FILE.exists():
        print(f"ERROR: {HYPOTHESES_FILE} not found", file=sys.stderr)
        sys.exit(1)

    hypotheses = load_hypotheses(HYPOTHESES_FILE)

    print(f"Generating curated examples → {OUT_FILE}")
    with open(OUT_FILE, "w") as fout:
        for ex in EXAMPLES:
            features_meta = select_features(hypotheses, ex["cwe"], TOP_K)
            top_features = make_activations(features_meta, bias="vulnerable")

            record = {
                "vuln_id": ex["id"],
                "cwe": ex["cwe"],
                "title": ex["title"],
                "file_extension": ex["file_extension"],
                "secure_code": ex["secure_code"],
                "vulnerable_code": ex["vulnerable_code"],
                "top_features": top_features,
                "placeholder_activations": True,  # flag to replace after real inference
            }
            fout.write(json.dumps(record) + "\n")

    print(f"  Written {len(EXAMPLES)} examples to {OUT_FILE}")
    print("Done. Run real SAE inference to replace placeholder activations.")


if __name__ == "__main__":
    main()
