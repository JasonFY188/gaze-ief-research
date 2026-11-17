# Ideas to think about later (sotsuron, gaze cone)

## 1. Pick the smallest cone shape per photo (noted 2026-11-16)

**Idea.** For every photo, compute the circle (95%), the ellipse (95% joint) and the ellipse (99% each),
and use whichever is smallest.

**Things to decide first**
- **Compare at the same coverage.** Only shapes that hold the same share of L2CS guesses are fair to compare.
  - Circle 95% and ellipse 95% joint both hold 95% -> fair to pick the smaller one.
  - Ellipse 99% each holds ~96-97% (more) -> it may look bigger simply because it covers more.
    Either compare it with a circle 97%, or drop it from the "pick smallest" set.
  - Ellipse 95% each holds only ~84% -> never compare it directly with the 95% circle.
- **"Smallest" in which units?** Area in (yaw, pitch) degrees? Or the solid angle (true size on the sphere)?
  Or the pixel area of the projected mask in view2 (what DeepGaze actually sees)? They can disagree.
- **Choosing per photo uses the photo's own L2CS output only** (no ground truth), so it is allowed at test time.
  Must not choose using the GT or the AUC (that would be cheating).

**Bigger version of the same idea.** The smallest possible region holding 95% of the guesses is the
"highest-density region" of the L2CS yaw x pitch distribution (keep the most likely cells until 95% is reached).
Circle and ellipse are just simple shapes approximating it. Worth trying as an upper bound.

**First data point (person_01.jpg, Person1):** yaw +-12, pitch +-9 -> circle95 r=12.5;
ellipse95 joint area = 0.99 x circle (no gain); ellipse99 each = 1.14 x circle. The ellipse only helps when
yaw and pitch uncertainty differ a lot; check how often that happens over all photos first.

**To do**
1. Run all experiment photos; look at the distribution of yaw_half / pitch_half (Excel columns).
2. If the ratio is often > 1.5, try "pick smallest at equal coverage"; else the ellipse is not worth it.
3. Report coverage next to every size, always.
