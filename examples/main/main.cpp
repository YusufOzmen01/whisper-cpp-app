#include "common.h"
#include "httplib.h"
#include "whisper.h"
#include "grammar-parser.h"
#include "base64.h"

#include <fstream>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <cstring>
#include <json.hpp>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// command-line parameters
struct whisper_params {
    int32_t n_threads     = std::min(4, static_cast<int32_t>(std::thread::hardware_concurrency()));
    int32_t n_processors  = 1;
    int32_t offset_t_ms   = 0;
    int32_t offset_n      = 0;
    int32_t duration_ms   = 0;
    int32_t progress_step = 5;
    int32_t max_context   = -1;
    int32_t max_len       = 0;
    int32_t best_of       = whisper_full_default_params(WHISPER_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size     = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
    int32_t audio_ctx     = 0;

    float word_thold      =  0.01f;
    float entropy_thold   =  2.40f;
    float logprob_thold   = -1.00f;
    float grammar_penalty = 100.0f;
    float temperature     = 0.0f;
    float temperature_inc = 0.2f;

    bool debug_mode      = false;
    bool translate       = false;
    bool detect_language = false;
    bool split_on_word   = false;
    bool no_fallback     = false;
    bool output_txt      = false;
    bool output_jsn      = false;
    bool output_jsn_full = false;
    bool log_score       = false;
    bool use_gpu         = true;
    bool flash_attn      = false;

    std::string language  = "en";
    std::string prompt;
    std::string model     = "models/ggml-base.en.bin";
    std::string grammar;
    std::string grammar_rule;

    std::string tdrz_speaker_turn = "[SPEAKER_TURN]"; // TODO: set from command line

    // A regular expression that matches tokens to suppress
    std::string suppress_regex;

    std::string openvino_encode_device = "CPU";

    std::string dtw;

    grammar_parser::parse_state grammar_parsed;
};

static std::string estimate_diarization_speaker(const std::vector<std::vector<float>> &pcmf32s, int64_t t0, int64_t t1, bool id_only = false) {
    std::string speaker;
    const int64_t n_samples = pcmf32s[0].size();

    const int64_t is0 = timestamp_to_sample(t0, n_samples, WHISPER_SAMPLE_RATE);
    const int64_t is1 = timestamp_to_sample(t1, n_samples, WHISPER_SAMPLE_RATE);

    double energy0 = 0.0f;
    double energy1 = 0.0f;

    for (int64_t j = is0; j < is1; j++) {
        energy0 += fabs(pcmf32s[0][j]);
        energy1 += fabs(pcmf32s[1][j]);
    }

    if (energy0 > 1.1*energy1) {
        speaker = "0";
    } else if (energy1 > 1.1*energy0) {
        speaker = "1";
    } else {
        speaker = "?";
    }

    //printf("is0 = %lld, is1 = %lld, energy0 = %f, energy1 = %f, speaker = %s\n", is0, is1, energy0, energy1, speaker.c_str());

    if (!id_only) {
        speaker.insert(0, "(speaker ");
        speaker.append(")");
    }

    return speaker;
}

static char * escape_double_quotes_and_backslashes(const char * str) {
    if (str == nullptr) {
        return nullptr;
    }

    size_t escaped_length = strlen(str) + 1;

    for (size_t i = 0; str[i] != '\0'; i++) {
        if (str[i] == '"' || str[i] == '\\') {
            escaped_length++;
        }
    }

    char * escaped = static_cast<char *>(calloc(escaped_length, 1)); // pre-zeroed
    if (escaped == nullptr) {
        return nullptr;
    }

    size_t pos = 0;
    for (size_t i = 0; str[i] != '\0'; i++) {
        if (str[i] == '"' || str[i] == '\\') {
            escaped[pos++] = '\\';
        }
        escaped[pos++] = str[i];
    }

    // no need to set zero due to calloc() being used prior

    return escaped;
}

std::string output_json(whisper_context * ctx, const std::vector<std::vector<float>>&  pcmf32s, bool full) {
    std::string out;

    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);

        out += text + std::string(" ");
    }

    return out;
}

static void cb_log_disable(enum ggml_log_level , const char * , void * ) { }

whisper_context* init_model(std::string model_path, std::string language) {
    whisper_params params = {};
    params.model = std::move(model_path);
    params.language = std::move(language);

    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());

        return nullptr;
    }

    whisper_log_set(cb_log_disable, nullptr);

    // whisper init

    whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    if (!params.dtw.empty()) {
        cparams.dtw_token_timestamps = true;
        cparams.dtw_aheads_preset = WHISPER_AHEADS_NONE;

        if (params.dtw == "tiny")      cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY;
        if (params.dtw == "tiny.en")   cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY_EN;
        if (params.dtw == "base")      cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE;
        if (params.dtw == "base.en")   cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE_EN;
        if (params.dtw == "small")     cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL;
        if (params.dtw == "small.en")  cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL_EN;
        if (params.dtw == "medium")    cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM;
        if (params.dtw == "medium.en") cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM_EN;
        if (params.dtw == "large.v1")  cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V1;
        if (params.dtw == "large.v2")  cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V2;
        if (params.dtw == "large.v3")  cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V3;
        if (params.dtw == "large.v3.turbo")  cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V3_TURBO;

        if (cparams.dtw_aheads_preset == WHISPER_AHEADS_NONE) {
            fprintf(stderr, "error: unknown DTW preset '%s'\n", params.dtw.c_str());

            return nullptr;
        }
    }

    whisper_context* ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");

        return nullptr;
    }

    // initialize openvino encoder. this has no effect on whisper.cpp builds that don't have OpenVINO configured
    whisper_ctx_init_openvino_encoder(ctx, nullptr, params.openvino_encode_device.c_str(), nullptr);

    if (!params.grammar.empty()) {
        auto & grammar = params.grammar_parsed;
        if (is_file_exist(params.grammar.c_str())) {
            // read grammar from file
            std::ifstream ifs(params.grammar.c_str());
            const std::string txt = std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            grammar = grammar_parser::parse(txt.c_str());
        } else {
            // read grammar from string
            grammar = grammar_parser::parse(params.grammar.c_str());
        }

        // will be empty (default) if there are parse errors
        if (grammar.rules.empty()) {
            fprintf(stderr, "error: failed to parse grammar \"%s\"\n", params.grammar.c_str());

            return nullptr;
        } else {
            fprintf(stderr, "%s: grammar:\n", __func__);
            grammar_parser::print_grammar(stderr, grammar);
            fprintf(stderr, "\n");
        }
    }

    return ctx;
}

std::string run_detection(const std::string& b64_in, std::string lang, whisper_context* ctx) {
    whisper_params params = {};
    params.language = std::move(lang);

    std::vector<float> pcmf32;               // mono-channel F32 PCM
    std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

    std::string buf = base64_decode(b64_in);
    if (!::read_wav(buf, pcmf32, pcmf32s, false)) {
        fprintf(stderr, "error: failed to read WAV file\n");

        return "";
    }

    // run the inference
    {
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

        const bool use_grammar = (!params.grammar_parsed.rules.empty() && !params.grammar_rule.empty());
        wparams.strategy = (params.beam_size > 1 || use_grammar) ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

        wparams.print_realtime   = false;
        wparams.translate        = params.translate;
        wparams.language         = params.language.c_str();
        wparams.detect_language  = params.detect_language;
        wparams.n_threads        = params.n_threads;
        wparams.offset_ms        = params.offset_t_ms;
        wparams.duration_ms      = params.duration_ms;

        wparams.thold_pt         = params.word_thold;
        wparams.max_len          = params.max_len;
        wparams.split_on_word    = params.split_on_word;
        wparams.audio_ctx        = params.audio_ctx;

        wparams.debug_mode       = params.debug_mode;

        wparams.suppress_regex   = params.suppress_regex.empty() ? nullptr : params.suppress_regex.c_str();

        wparams.initial_prompt   = params.prompt.c_str();

        wparams.greedy.best_of        = params.best_of;
        wparams.beam_search.beam_size = params.beam_size;

        wparams.temperature      = params.temperature;

        wparams.entropy_thold    = params.entropy_thold;
        wparams.logprob_thold    = params.logprob_thold;

        const auto & grammar_parsed = params.grammar_parsed;
        auto grammar_rules = grammar_parsed.c_rules();

        if (use_grammar) {
            if (grammar_parsed.symbol_ids.find(params.grammar_rule) == grammar_parsed.symbol_ids.end()) {
                fprintf(stderr, "%s: warning: grammar rule '%s' not found - skipping grammar sampling\n", __func__, params.grammar_rule.c_str());
            } else {
                wparams.grammar_rules = grammar_rules.data();
                wparams.n_grammar_rules = grammar_rules.size();
                wparams.i_start_rule = grammar_parsed.symbol_ids.at(params.grammar_rule);
                wparams.grammar_penalty = params.grammar_penalty;
            }
        }

        // examples for abort mechanism
        // in examples below, we do not abort the processing, but we could if the flag is set to true

        // the callback is called before every encoder run - if it returns false, the processing is aborted
        {
            static bool is_aborted = false; // NOTE: this should be atomic to avoid data race

            wparams.encoder_begin_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, void * user_data) {
                bool is_aborted = *static_cast<bool *>(user_data);
                return !is_aborted;
            };
            wparams.encoder_begin_callback_user_data = &is_aborted;
        }

        // the callback is called before every computation - if it returns true, the computation is aborted
        {
            static bool is_aborted = false; // NOTE: this should be atomic to avoid data race

            wparams.abort_callback = [](void * user_data) {
                bool is_aborted = *static_cast<bool *>(user_data);
                return is_aborted;
            };
            wparams.abort_callback_user_data = &is_aborted;
        }

        if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0) {
            fprintf(stderr, "failed to process audio\n");

            return "";
        }
    }

    return output_json(ctx, pcmf32s, params.output_jsn_full);
}

whisper_context* ctx = nullptr;

int main() {
    httplib::Server svr;

    svr.Post("/init_model", [](const httplib::Request &req, httplib::Response &res) {
        auto json = nlohmann::json::parse(req.body);

        if (json.is_null()) {
            res.status = 400;
            res.set_content("Invalid json", "text/plain");

            return;
        }

        if (json["modelpath"] == nullptr) {
            res.status = 400;
            res.set_content("Invalid model path", "text/plain");

            return;
        }

        if (json["lang"] == nullptr) {
            res.status = 400;
            res.set_content("Invalid lang", "text/plain");

            return;
        }

        if (ctx != nullptr) {
            whisper_free(ctx);

            ctx = nullptr;
        }

        ctx = init_model(json["modelpath"].get<std::string>(), json["lang"].get<std::string>());

        if (ctx == nullptr) {
            res.status = 500;
            res.set_content("Could not initialize model. Maybe model path is wrong?", "text/plain");
        }
    });

    svr.Post("/run_detection", [](const httplib::Request &req, httplib::Response &res) {
        auto json = nlohmann::json::parse(req.body);

        if (json.is_null()) {
            res.status = 400;
            res.set_content("Invalid json", "text/plain");

            return;
        }

        if (json["lang"] == nullptr) {
            res.status = 400;
            res.set_content("Invalid model path", "text/plain");

            return;
        }

        if (json["wavdata"] == nullptr) {
            res.status = 400;
            res.set_content("Invalid wavdata", "text/plain");

            return;
        }

        auto data = run_detection(json["wavdata"].get<std::string>(), json["lang"].get<std::string>(), ctx);
        if (data.size() == 0) {
            res.status = 500;
            res.set_content("Error occured while detecting speech", "text/plain");

            return;
        }

        res.set_content(data, "plain/text");
    });

    svr.listen("0.0.0.0", 8080);
}