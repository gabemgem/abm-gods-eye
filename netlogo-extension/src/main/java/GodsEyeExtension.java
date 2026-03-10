import org.nlogo.api.*;
import org.nlogo.core.Syntax;
import org.nlogo.core.SyntaxJ;

import java.io.*;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Map;

/**
 * ClassManager for the gods-eye NetLogo extension.
 *
 * Lifecycle:
 *   1. NetLogo loads the extension → load() registers primitives.
 *   2. Model calls gods-eye:init "provider" "model" → starts the Python
 *      bridge server as a subprocess and creates a session.
 *   3. Model calls gods-eye:ask / gods-eye:record-state during its run.
 *   4. Model closes → unload() kills the subprocess.
 *
 * The Python server is started lazily on the first gods-eye:init call
 * so that the API key, provider, and model are all known before launch.
 */
public class GodsEyeExtension extends DefaultClassManager {

    // Shared state accessed by all primitives via GodsEyeExtension.state()
    static final ServerState STATE = new ServerState();

    @Override
    public void load(PrimitiveManager pm) {
        pm.addPrimitive("init",         new InitCommand());
        pm.addPrimitive("ask",          new AskReporter());
        pm.addPrimitive("record-state", new RecordStateCommand());
        pm.addPrimitive("end-session",  new EndSessionCommand());
    }

    @Override
    public void unload(ExtensionManager em) {
        STATE.shutdown();
    }

    // ------------------------------------------------------------------
    // Shared HTTP client (re-used across calls)
    // ------------------------------------------------------------------
    static final HttpClient HTTP = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .build();

    // ------------------------------------------------------------------
    // Inner class: mutable server state shared by primitives
    // ------------------------------------------------------------------
    static class ServerState {
        volatile Process   serverProcess = null;
        volatile String    sessionId     = null;
        volatile int       port          = 8765;
        volatile boolean   ready         = false;

        String baseUrl() {
            return "http://127.0.0.1:" + port;
        }

        void shutdown() {
            try {
                if (sessionId != null) {
                    httpDelete(baseUrl() + "/session/" + sessionId);
                    sessionId = null;
                }
            } catch (Exception ignored) {}

            if (serverProcess != null && serverProcess.isAlive()) {
                serverProcess.destroy();
                serverProcess = null;
            }
            ready = false;
        }
    }

    // ------------------------------------------------------------------
    // HTTP helpers
    // ------------------------------------------------------------------

    /** POST JSON body, return response body string. */
    static String httpPost(String url, String jsonBody) throws ExtensionException {
        try {
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .timeout(Duration.ofSeconds(120))
                    .build();
            HttpResponse<String> resp = HTTP.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() >= 400) {
                throw new ExtensionException("gods-eye server error " + resp.statusCode() + ": " + resp.body());
            }
            return resp.body();
        } catch (ExtensionException e) {
            throw e;
        } catch (Exception e) {
            throw new ExtensionException("gods-eye HTTP error: " + e.getMessage());
        }
    }

    /** DELETE request. */
    static void httpDelete(String url) throws Exception {
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .DELETE()
                .timeout(Duration.ofSeconds(10))
                .build();
        HTTP.send(req, HttpResponse.BodyHandlers.ofString());
    }

    /** GET request, return body. */
    static String httpGet(String url) throws Exception {
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .GET()
                .timeout(Duration.ofSeconds(10))
                .build();
        HttpResponse<String> resp = HTTP.send(req, HttpResponse.BodyHandlers.ofString());
        return resp.body();
    }

    /**
     * Poll /health until the server responds OK or timeout elapses.
     *
     * @param timeoutMs maximum wait in milliseconds
     */
    static void waitForServer(String baseUrl, long timeoutMs) throws ExtensionException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        String healthUrl = baseUrl + "/health";
        while (System.currentTimeMillis() < deadline) {
            try {
                String body = httpGet(healthUrl);
                if (body.contains("ok")) return;
            } catch (Exception ignored) {}
            try { Thread.sleep(250); } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                throw new ExtensionException("Interrupted while waiting for gods-eye server");
            }
        }
        throw new ExtensionException(
                "gods-eye server did not start within " + (timeoutMs / 1000) + "s. " +
                "Check that Python and abm-gods-eye[server] are installed.");
    }
}
