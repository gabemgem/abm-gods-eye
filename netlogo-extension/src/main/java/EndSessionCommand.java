import org.nlogo.api.*;
import org.nlogo.core.Syntax;
import org.nlogo.core.SyntaxJ;

/**
 * gods-eye:end-session
 *
 * Explicitly ends the current conversation session, clearing the LLM's
 * conversation history on the server. Useful if you want to start a
 * fresh conversation mid-run without restarting the server.
 *
 * The Python server subprocess keeps running; call gods-eye:init again
 * to open a new session with the same or different provider/model.
 *
 * Example:
 *   to reset-observer
 *     gods-eye:end-session
 *     gods-eye:init "anthropic" "claude-sonnet-4-6"
 *   end
 */
public class EndSessionCommand extends DefaultCommand {

    @Override
    public Syntax getSyntax() {
        return SyntaxJ.commandSyntax(new int[]{});
    }

    @Override
    public void perform(Argument[] args, Context ctx) throws ExtensionException {
        GodsEyeExtension.ServerState state = GodsEyeExtension.STATE;

        if (state.sessionId == null) return; // nothing to end

        try {
            GodsEyeExtension.httpDelete(state.baseUrl() + "/session/" + state.sessionId);
        } catch (Exception e) {
            throw new ExtensionException("gods-eye: failed to end session: " + e.getMessage());
        } finally {
            state.sessionId = null;
            state.ready = false;
        }
    }
}
