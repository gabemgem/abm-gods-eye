import org.nlogo.api.*;
import org.nlogo.core.Syntax;
import org.nlogo.core.SyntaxJ;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

/**
 * gods-eye:ask question → string
 *
 * Sends a natural-language question to the GodsEye bridge server and
 * returns the LLM's response as a NetLogo string.
 *
 * Conversation history is preserved across calls within the same session,
 * so the observer remembers what was discussed earlier.
 *
 * Example:
 *   show gods-eye:ask "How many agents are happy right now?"
 *   set answer gods-eye:ask "What trends have you noticed?"
 */
public class AskReporter extends DefaultReporter {

    private static final Gson GSON = new Gson();

    @Override
    public Syntax getSyntax() {
        return SyntaxJ.reporterSyntax(
                new int[]{Syntax.StringType()},
                Syntax.StringType()
        );
    }

    @Override
    public Object report(Argument[] args, Context ctx) throws ExtensionException {
        GodsEyeExtension.ServerState state = GodsEyeExtension.STATE;

        if (!state.ready || state.sessionId == null) {
            throw new ExtensionException(
                    "gods-eye: not initialised. Call gods-eye:init before gods-eye:ask.");
        }

        String question;
        try {
            question = args[0].getString();
        } catch (LogoException e) {
            throw new ExtensionException(e.getMessage());
        }

        // Build request body: {"question": "<question>"}
        JsonObject body = new JsonObject();
        body.addProperty("question", question);
        String jsonBody = GSON.toJson(body);

        String url = state.baseUrl() + "/session/" + state.sessionId + "/ask";
        String responseJson = GodsEyeExtension.httpPost(url, jsonBody);

        // Parse {"response": "..."}
        JsonObject parsed = GSON.fromJson(responseJson, JsonObject.class);
        if (parsed == null || !parsed.has("response")) {
            throw new ExtensionException("gods-eye: unexpected response from server: " + responseJson);
        }
        return parsed.get("response").getAsString();
    }
}
