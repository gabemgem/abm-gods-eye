import org.nlogo.api.*;
import org.nlogo.core.Syntax;
import org.nlogo.core.SyntaxJ;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonArray;

/**
 * gods-eye:record-state
 *
 * Pushes the current NetLogo world state to the bridge server as a snapshot.
 * Call this once per tick (inside your "go" procedure) to build up history
 * that the LLM can reference when answering trend/comparison questions.
 *
 * The command automatically reads:
 *   - Current tick number
 *   - All observer (global) variables
 *   - All turtle breeds and their variables (up to 200 turtles)
 *   - Aggregate counts per breed
 *
 * Example usage in NetLogo:
 *   to go
 *     ask turtles [ move ]
 *     gods-eye:record-state
 *     tick
 *   end
 */
public class RecordStateCommand extends DefaultCommand {

    private static final Gson GSON = new Gson();

    @Override
    public Syntax getSyntax() {
        return SyntaxJ.commandSyntax(new int[]{});
    }

    @Override
    public void perform(Argument[] args, Context ctx) throws ExtensionException {
        GodsEyeExtension.ServerState state = GodsEyeExtension.STATE;

        if (!state.ready || state.sessionId == null) {
            throw new ExtensionException(
                    "gods-eye: not initialised. Call gods-eye:init before gods-eye:record-state.");
        }

        org.nlogo.api.World world = ctx.world();
        JsonObject snapshot = buildSnapshot(world);

        JsonObject body = new JsonObject();
        body.add("state", snapshot);
        String jsonBody = GSON.toJson(body);

        String url = state.baseUrl() + "/session/" + state.sessionId + "/snapshot";
        GodsEyeExtension.httpPost(url, jsonBody);
    }

    // ------------------------------------------------------------------
    // World serialisation
    // ------------------------------------------------------------------

    private static JsonObject buildSnapshot(org.nlogo.api.World world) {
        JsonObject snap = new JsonObject();

        // Tick
        snap.addProperty("tick", (long) world.ticks());

        // Observer (global) variables
        JsonObject globals = new JsonObject();
        String[] varNames = world.observerVariables();
        if (varNames != null) {
            for (String name : varNames) {
                try {
                    Object val = world.getObserverVariableByName(name);
                    addLogoValue(globals, name.toLowerCase(), val);
                } catch (Exception ignored) {}
            }
        }
        snap.add("globals", globals);

        // Aggregate turtle counts per breed
        JsonObject breedCounts = new JsonObject();
        int totalTurtles = 0;
        try {
            org.nlogo.api.AgentSet allTurtles = world.turtles();
            if (allTurtles != null) {
                totalTurtles = allTurtles.count();
            }
        } catch (Exception ignored) {}
        snap.addProperty("total_turtles", totalTurtles);

        // Turtle snapshots (capped at 200)
        JsonArray agentArray = new JsonArray();
        try {
            org.nlogo.api.AgentSet allTurtles = world.turtles();
            if (allTurtles != null) {
                int count = 0;
                for (org.nlogo.api.Agent agent : allTurtles.agents()) {
                    if (count >= 200) break;
                    if (agent instanceof org.nlogo.api.Turtle) {
                        org.nlogo.api.Turtle t = (org.nlogo.api.Turtle) agent;
                        agentArray.add(turtleToJson(t));
                        count++;
                    }
                }
            }
        } catch (Exception ignored) {}
        snap.add("agents", agentArray);

        return snap;
    }

    private static JsonObject turtleToJson(org.nlogo.api.Turtle t) {
        JsonObject obj = new JsonObject();
        try { obj.addProperty("who", t.id()); } catch (Exception ignored) {}
        try { obj.addProperty("breed", t.getBreed().printName().toLowerCase()); } catch (Exception ignored) {}
        try { obj.addProperty("xcor", t.xcor()); } catch (Exception ignored) {}
        try { obj.addProperty("ycor", t.ycor()); } catch (Exception ignored) {}
        try { obj.addProperty("heading", t.heading()); } catch (Exception ignored) {}
        try { obj.addProperty("color", t.color()); } catch (Exception ignored) {}
        try { obj.addProperty("size", t.size()); } catch (Exception ignored) {}

        // Turtle-own variables
        try {
            String[] varNames = t.variableNames();
            if (varNames != null) {
                for (int i = 0; i < varNames.length; i++) {
                    String name = varNames[i].toLowerCase();
                    // Skip built-in variables already captured above
                    if (isBuiltinTurtleVar(name)) continue;
                    try {
                        Object val = t.getVariable(i);
                        addLogoValue(obj, name, val);
                    } catch (Exception ignored) {}
                }
            }
        } catch (Exception ignored) {}

        return obj;
    }

    private static boolean isBuiltinTurtleVar(String name) {
        return name.equals("who") || name.equals("color") || name.equals("heading") ||
               name.equals("xcor") || name.equals("ycor") || name.equals("shape") ||
               name.equals("label") || name.equals("label-color") || name.equals("breed") ||
               name.equals("hidden?") || name.equals("size") || name.equals("pen-size") ||
               name.equals("pen-mode");
    }

    /** Add a NetLogo value to a JsonObject, mapping to the nearest JSON type. */
    private static void addLogoValue(JsonObject obj, String key, Object val) {
        if (val instanceof Double)       obj.addProperty(key, (Double) val);
        else if (val instanceof Boolean) obj.addProperty(key, (Boolean) val);
        else if (val instanceof String)  obj.addProperty(key, (String) val);
        else if (val == null)            obj.add(key, com.google.gson.JsonNull.INSTANCE);
        else                             obj.addProperty(key, val.toString());
    }
}
