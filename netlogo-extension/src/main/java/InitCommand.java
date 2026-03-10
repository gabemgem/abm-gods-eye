import org.nlogo.api.*;
import org.nlogo.core.Syntax;
import org.nlogo.core.SyntaxJ;

import java.io.*;
import java.util.*;

/**
 * gods-eye:init provider model
 * gods-eye:init provider model port        (optional third arg)
 * gods-eye:init provider model port env-file  (optional fourth arg)
 *
 * Examples:
 *   gods-eye:init "anthropic" "claude-sonnet-4-6"
 *   gods-eye:init "openai" "gpt-4o"
 *   gods-eye:init "google" "gemini-2.0-flash" 8765 "/path/to/.env"
 *
 * Starts the Python bridge server as a subprocess (if not already running),
 * waits for it to become healthy, then opens a conversation session.
 *
 * API keys are read from the .env file or environment — never passed as args.
 */
public class InitCommand extends DefaultCommand {

    @Override
    public Syntax getSyntax() {
        // Required: provider, model  (strings)
        // Optional: port (number), env-file path (string)
        return SyntaxJ.commandSyntax(new int[]{
                Syntax.StringType(),
                Syntax.StringType(),
                Syntax.NumberType() | Syntax.RepeatableType()
        }, 2);
    }

    @Override
    public void perform(Argument[] args, Context ctx) throws ExtensionException {
        GodsEyeExtension.ServerState state = GodsEyeExtension.STATE;

        // Shut down any previously running server / session cleanly
        if (state.ready) {
            state.shutdown();
        }

        String provider = arg(args, 0);
        String model    = arg(args, 1);
        int    port     = args.length > 2 ? (int) argDouble(args, 2) : 8765;
        String envFile  = args.length > 3 ? arg(args, 3) : ".env";

        state.port = port;

        // ------------------------------------------------------------------
        // Locate python executable
        // ------------------------------------------------------------------
        String python = findPython();

        // ------------------------------------------------------------------
        // Build subprocess environment — pass provider/model/port via env vars
        // so the server can read them, but never touch API keys directly.
        // ------------------------------------------------------------------
        ProcessBuilder pb = new ProcessBuilder(
                python, "-m", "abm_gods_eye.server",
                "--provider", provider,
                "--model",    model,
                "--port",     String.valueOf(port),
                "--env-file", envFile
        );

        // Inherit the parent environment (picks up API keys already set in shell)
        pb.environment().put("GODS_EYE_PROVIDER", provider);
        pb.environment().put("GODS_EYE_MODEL",    model);
        pb.environment().put("GODS_EYE_PORT",     String.valueOf(port));

        // Redirect stderr to a log file alongside the model; stdout is silent
        pb.redirectErrorStream(false);
        pb.redirectOutput(ProcessBuilder.Redirect.DISCARD);
        pb.redirectError(new File("gods-eye-server.log"));

        try {
            state.serverProcess = pb.start();
        } catch (IOException e) {
            throw new ExtensionException(
                    "Failed to start Python server. Is '" + python + "' on your PATH " +
                    "and is abm-gods-eye[server] installed?\nError: " + e.getMessage());
        }

        // ------------------------------------------------------------------
        // Wait up to 30 s for the server to become healthy
        // ------------------------------------------------------------------
        GodsEyeExtension.waitForServer(state.baseUrl(), 30_000);

        // ------------------------------------------------------------------
        // Open a session
        // ------------------------------------------------------------------
        String resp = GodsEyeExtension.httpPost(
                state.baseUrl() + "/session/start",
                "{}"
        );

        // Parse session_id from JSON: {"session_id":"<uuid>"}
        state.sessionId = extractStringField(resp, "session_id");
        state.ready = true;
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static String arg(Argument[] args, int i) throws ExtensionException {
        try { return args[i].getString(); }
        catch (LogoException e) { throw new ExtensionException(e.getMessage()); }
    }

    private static double argDouble(Argument[] args, int i) throws ExtensionException {
        try { return args[i].getDoubleValue(); }
        catch (LogoException e) { throw new ExtensionException(e.getMessage()); }
    }

    /**
     * Find the Python executable, preferring the same interpreter that would
     * run on the user's PATH. Tries "python3" then "python".
     */
    private static String findPython() {
        for (String candidate : new String[]{"python3", "python"}) {
            try {
                Process p = new ProcessBuilder(candidate, "--version")
                        .redirectErrorStream(true)
                        .start();
                int exit = p.waitFor();
                if (exit == 0) return candidate;
            } catch (Exception ignored) {}
        }
        return "python"; // fall back; will fail with a clear OS error
    }

    /** Minimal JSON string-field extractor — avoids bundling a full parser just for this. */
    static String extractStringField(String json, String field) throws ExtensionException {
        String needle = "\"" + field + "\"";
        int idx = json.indexOf(needle);
        if (idx < 0) throw new ExtensionException("gods-eye: field '" + field + "' not found in: " + json);
        int colon = json.indexOf(':', idx + needle.length());
        int open  = json.indexOf('"', colon + 1);
        int close = json.indexOf('"', open + 1);
        if (open < 0 || close < 0) throw new ExtensionException("gods-eye: malformed JSON: " + json);
        return json.substring(open + 1, close);
    }
}
