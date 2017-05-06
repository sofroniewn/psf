# psf

> Compute the point spread function from a 3d image of beads

This package computes the point spread function (psf) from a 3d image of beads. It first finds the centers of each well separated bead and then fits 2d Gaussians to max projections of the bead. It returns the resulting psfs as a table. Look at the examply jupyter notebook for usage instructions.

### install

```bash
git clone https://github.com/sofroniewn/psf
```
