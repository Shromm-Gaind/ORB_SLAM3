// parity_dump.cc — emit the C++ DormantTrackBuffer's behaviour on a
// fixed operation script as CSV, so it can be diffed against the Python
// reference (dormant_buffer.py) run through the same script.
//
// Parity here is behavioural, not by inspection: if the two
// implementations ever diverge on purge boundaries, L-infinity edges or
// motion-compensation accumulation, this diff catches it.

#include "DormantTrackBuffer.h"
#include "SpatialDescriptorMatcher.h"
#include <cstdio>

using namespace hybrid_frontend;

static Descriptor256 bits(int n) {
    Descriptor256 d{};
    d.fill(0);
    for (int i = 0; i < n; ++i)
        d[i / 8] |= static_cast<std::uint8_t>(1u << (i % 8));
    return d;
}

static DormantTrack mk(std::uint64_t id, float x, float y,
                       std::uint64_t died, int octave, std::uint32_t age) {
    DormantTrack t;
    t.id = id; t.last_x = x; t.last_y = y; t.frame_died = died;
    t.octave = octave; t.age_at_death = age;
    t.descriptor.fill(static_cast<std::uint8_t>(id & 0xFF));
    t.map_point = nullptr;
    return t;
}

static void dump(const char* tag, const DormantTrackBuffer& b) {
    printf("%s|size=%zu", tag, b.size());
    for (const auto& e : b.all_entries())
        printf("|%llu:%.2f,%.2f:d%llu:o%d:a%u",
               (unsigned long long)e.id, e.last_x, e.last_y,
               (unsigned long long)e.frame_died, e.octave, e.age_at_death);
    printf("\n");
}

static void dump_query(const char* tag, const DormantTrackBuffer& b,
                       float x, float y, float r) {
    auto hits = b.query_within(x, y, r);
    printf("%s|n=%zu", tag, hits.size());
    for (const auto& h : hits) printf("|%llu", (unsigned long long)h.id);
    printf("\n");
}

int main() {
    DormantTrackBuffer buf(30);

    buf.add(mk(1, 100.0f, 100.0f, 100, 0, 5));
    buf.add(mk(2, 110.0f, 100.0f, 100, 2, 40));
    buf.add(mk(3, 110.0f, 110.0f, 105, 5, 300));
    buf.add(mk(4, 500.0f, 500.0f, 120, 7, 1));
    dump("after_adds", buf);

    // L-infinity edge cases: exact edge, corner, just-outside.
    dump_query("q_r10", buf, 100.0f, 100.0f, 10.0f);
    dump_query("q_r0", buf, 100.0f, 100.0f, 0.0f);
    dump_query("q_rneg", buf, 100.0f, 100.0f, -5.0f);
    dump_query("q_r9_99", buf, 100.0f, 100.0f, 9.99f);

    // Motion compensation, then the same queries.
    buf.translate_all(5.0f, -2.0f);
    dump("after_translate", buf);
    dump_query("q_after_tr", buf, 105.0f, 98.0f, 10.0f);

    buf.translate_all(0.0f, 0.0f);   // no-op
    buf.translate_all(-5.0f, 2.0f);  // undo
    dump("after_translate_undo", buf);

    // Purge boundaries: underflow, exact horizon, one beyond.
    buf.purge_older_than(5);
    dump("purge_underflow", buf);
    buf.purge_older_than(130);   // cutoff 100 -> id1,id2 (died 100) kept
    dump("purge_at_horizon", buf);
    buf.purge_older_than(131);   // cutoff 101 -> id1,id2 dropped
    dump("purge_beyond", buf);
    buf.purge_older_than(136);   // cutoff 106 -> id3 (died 105) dropped
    dump("purge_more", buf);

    printf("remove_hit=%d\n", (int)buf.remove(4));
    printf("remove_miss=%d\n", (int)buf.remove(999));
    dump("after_remove", buf);

    buf.add(mk(9, 10.0f, 10.0f, 140, 1, 7));
    buf.clear();
    dump("after_clear", buf);

    // gap_frames, including the backwards-clock guard.
    auto t = mk(1, 0.0f, 0.0f, 100, 0, 0);
    printf("gap|%llu|%llu|%llu\n",
           (unsigned long long)DormantTrackBuffer::gap_frames(t, 100),
           (unsigned long long)DormantTrackBuffer::gap_frames(t, 130),
           (unsigned long long)DormantTrackBuffer::gap_frames(t, 50));
    // ---- SpatialDescriptorMatcher parity --------------------------
    // One scenario per gate, all in a single call so ordering and the
    // unique-candidates pass are exercised together.
    {
        std::vector<PixelQuery> qs;
        auto Q = [&](float x, float y, int nb, float r) {
            PixelQuery q; q.x = x; q.y = y; q.descriptor = bits(nb);
            q.radius = r; qs.push_back(q);
        };
        // q0: plain best-of-three
        Q(100.0f, 100.0f, 0, -1.0f);
        // q1: per-query radius override reaches a far candidate
        Q(200.0f, 200.0f, 0, 40.0f);
        // q2: margin-gate ambiguity (two close candidates)
        Q(300.0f, 300.0f, 0, -1.0f);
        // q3: single candidate, trivially distinct
        Q(400.0f, 400.0f, 0, -1.0f);
        // q4: non-passing competitor inside margin
        Q(500.0f, 500.0f, 0, -1.0f);
        // q5, q6: unique-candidates collision on candidate 8
        Q(600.0f, 600.0f, 30, -1.0f);
        Q(601.0f, 600.0f, 10, -1.0f);

        std::vector<PixelCandidate> cs;
        auto C = [&](float x, float y, int nb, int thr) {
            PixelCandidate c; c.x = x; c.y = y; c.descriptor = bits(nb);
            c.hamming_threshold = thr; cs.push_back(c);
        };
        C(101.0f, 100.0f, 80, -1);   // 0: q0, worse
        C(102.0f, 100.0f, 20, -1);   // 1: q0, best
        C(103.0f, 100.0f, 60, -1);   // 2: q0, middle
        C(230.0f, 200.0f, 40, 90);   // 3: q1, only in override radius; own thr 90
        C(301.0f, 300.0f, 20, -1);   // 4: q2 ambiguous pair
        C(302.0f, 300.0f, 22, -1);   // 5: q2 ambiguous pair
        C(401.0f, 400.0f, 45, -1);   // 6: q3 lone candidate
        C(501.0f, 500.0f, 25, 10);   // 7: q4 competitor, fails own thr 10
        C(502.0f, 500.0f, 20, -1);   // 8: q4 best but ambiguous vs 7
        C(602.0f, 600.0f, 0, -1);    // 9: q5/q6 contested candidate
        C(650.0f, 600.0f, 200, -1);  // 10: spatial for q5/q6, far in Hamming

        MatchOptions o;
        o.default_radius = 20.0f;
        o.hamming_threshold = 100;
        o.second_best_margin = 15;
        o.unique_candidates = true;

        auto res = SpatialDescriptorMatch(qs, cs, o);
        for (std::size_t i = 0; i < res.size(); ++i) {
            if (res[i].has_value())
                printf("match%zu|%zu|%d\n", i, res[i]->candidate_index,
                       res[i]->hamming_distance);
            else
                printf("match%zu|none\n", i);
        }
    }
    return 0;
}