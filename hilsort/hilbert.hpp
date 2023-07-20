// Filename: hilbert.hpp

/* LICENSE
 *
 * This software is copyrighted by Rice University.  It may be freely copied,
 * modified, and redistributed, provided that the copyright notice is
 * preserved on all copies.
 *
 * There is no warranty or other guarantee of fitness for this software,
 * it is provided solely "as is".  Bug reports or fixes may be sent
 * to the author, who may or may not act on them as he desires.
 *
 * You may include this software in a program or other software product,
 * but must display the notice:
 *
 * Hilbert Curve implementation copyright 1998, Rice University
 *
 * in any place where the end-user would see your own copyright.
 *
 * If you modify this software, you should include a notice giving the
 * name of the person performing the modification, the date of modification,
 * and the reason for such modification.
 */

/* Revision history:

   July 1998: Initial release

   Sept 1998: Second release

   Dec 1998: Fixed bug in hilbert_c2i that allowed a shift by number of bits in
   bitmask to vaporize index, in last bit of the function.  Implemented
   hilbert_incr.

   August 1999: Added argument to hilbert_nextinbox so that you can, optionally,
   find the previous point along the curve to intersect the box, rather than the
   next point.

   Nov 1999: Defined fast bit-transpose function (fast, at least, if the number
   of bits is large), and reimplemented i2c and c2i in terms of it.  Collapsed
   loops in hilbert_cmp, with the intention of reusing the cmp code to compare
   more general bitstreams.

   Feb 2000: Implemented almost all the floating point versions of cmp, etc, so
   that coordinates expressed in terms of double-precision IEEE floating point
   can be ordered.  Still have to do next-in-box, though.

   Oct 2001: Learned that some arbitrary coding choices caused some routines
   to fail in one dimension, and changed those choices.

   version 2001-10-20-05:34

*/


// The hilbert.hpp is taken from hilbert.h & hilbert.c by Doug Moore
//
// Computes Hilbert space-filling curve coordinates, without
// recursion, from integer index, and vice versa, and other Hilbert-related
// calculations.  Also known as Pi-order or Peano scan.
// 
// Author:      Doug Moore
//              Dept. of Computational and Applied Math
//              Rice University
//              http://www.caam.rice.edu/~dougm
// Date:        Sun Feb 20 2000
// Copyright (c) 1998-2000, Rice University
//
// Acknowledgement:
// This implementation is based on the work of A. R. Butz ("Alternative
// Algorithm for Hilbert's Space-Filling Curve", IEEE Trans. Comp., April,
// 1971, pp 424-426) and its interpretation by Spencer W. Thomas, University
// of Michigan (http://www-personal.umich.edu/~spencer/Home.html) in his widely
// available C software.  While the implementation here differs considerably
// from his, the first two interfaces and the style of some comments are very
// much derived from his work. 
//

// 
// Contributor:
//    Yaser Afshar, July 2023
// 
// The current implementation contains the original C code by Doug Moore
// taken from the author's webpage http://www.tiac.net/~sw/2008/10/Hilbert/moore/.
//
// It is updated as a standalone C++ header from the original work of Doug Moore with
// no change to the main functionality and with the goal of ease of use and C++ portability
//

#pragma once

#ifndef HILBERT_HPP
#define HILBERT_HPP

#include <cstdint>
#include <cstring>

#if defined(__GLIBC__) || defined(__GLIBCXX__) || defined(__GNU_LIBRARY__) || defined(__ANDROID__)
#include <endian.h>
#else
#if defined(macintosh) || defined(Macintosh) || (defined(__APPLE__) && defined(__MACH__))
#include <machine/endian.h>
#elif defined(BSD) || defined(_SYSTYPE_BSD)
#if defined(__OpenBSD__)
#include <machine/endian.h>
#else
#include <sys/endian.h>
#endif
#endif
#endif

#if defined(__BYTE_ORDER)
#if defined(__BIG_ENDIAN) && (__BYTE_ORDER == __BIG_ENDIAN)
#define HILBERT_BIG_ENDIAN 1
#elif defined(__LITTLE_ENDIAN) && (__BYTE_ORDER == __LITTLE_ENDIAN)
#define HILBERT_LITTLE_ENDIAN 1
#else
#define HILBERT_UNKNOWN_ENDIANNESS 1
#endif
#elif defined(__BIG_ENDIAN) && !defined(__LITTLE_ENDIAN)
#define HILBERT_BIG_ENDIAN 1
#elif defined(__LITTLE_ENDIAN) && !defined(__BIG_ENDIAN)
#define HILBERT_LITTLE_ENDIAN 1
#elif defined(_BYTE_ORDER)
#if defined(_BIG_ENDIAN) && (_BYTE_ORDER == _BIG_ENDIAN)
#define HILBERT_BIG_ENDIAN 1
#elif defined(_LITTLE_ENDIAN) && (_BYTE_ORDER == _LITTLE_ENDIAN)
#define HILBERT_LITTLE_ENDIAN 1
#else
#define HILBERT_UNKNOWN_ENDIANNESS 1
#endif
#elif defined(_BIG_ENDIAN) && !defined(_LITTLE_ENDIAN)
#define HILBERT_BIG_ENDIAN 1
#elif defined(_LITTLE_ENDIAN) && !defined(_BIG_ENDIAN)
#define HILBERT_LITTLE_ENDIAN 1
#elif defined(__APPLE__)
#if defined(__BIG_ENDIAN__)
#define HILBERT_BIG_ENDIAN 1
#elif defined(__LITTLE_ENDIAN__)
#define HILBERT_LITTLE_ENDIAN 1
#else
#define HILBERT_UNKNOWN_ENDIANNESS 1
#endif
#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define HILBERT_LITTLE_ENDIAN 1
#else
#define HILBERT_UNKNOWN_ENDIANNESS 1
#endif

#if defined(HILBERT_UNKNOWN_ENDIANNESS)
#if defined(sparc)
#undef HILBERT_UNKNOWN_ENDIANNESS
#define HILBERT_BIG_ENDIAN 1
#endif
#endif

#if defined(HILBERT_UNKNOWN_ENDIANNESS)
#error "Unknown endianness!"
#endif

/* define the bitmask_t type as an integer of sufficient size */
using bitmask_t = std::uint64_t;
/* define the halfmask_t type as an integer of 1/2 the size of bitmask_t */
using halfmask_t = std::uint32_t;

#define adjust_rotation(rotation, nDims, bits)               \
    do                                                       \
    {                                                        \
        /* rotation = (rotation + 1 + ffs(bits)) % nDims; */ \
        bits &= -bits & nd1Ones;                             \
        while (bits)                                         \
            bits >>= 1, ++rotation;                          \
        if (++rotation >= nDims)                             \
            rotation -= nDims;                               \
    } while (0)

#define ones(T, k) ((((T)2) << (k - 1)) - 1)

#define rdbit(w, k) (((w) >> (k)) & 1)

#define rotateRight(arg, nRots, nDims) ((((arg) >> (nRots)) | ((arg) << ((nDims) - (nRots)))) & ones(bitmask_t, nDims))

#define rotateLeft(arg, nRots, nDims) ((((arg) << (nRots)) | ((arg) >> ((nDims) - (nRots)))) & ones(bitmask_t, nDims))

#if defined(HILBERT_BIG_ENDIAN)
#define whichByte(nBytes, y) (nBytes - 1 - y / 8)
#define setBytes(dst, pos, nBytes, val) std::memset(&dst[pos + 1], val, nBytes - pos - 1)
#else
#define whichByte(nBytes, y) (y / 8)
#define setBytes(dst, pos, nBytes, val) std::memset(&dst[0], val, pos)
#endif

static bitmask_t bitTranspose(unsigned int nDims, unsigned int nBits, bitmask_t inCoords)
{
    unsigned int const nDims1{nDims - 1};
    unsigned int inB{nBits};
    unsigned int utB;
    bitmask_t inFieldEnds{1};
    bitmask_t inMask = ones(bitmask_t, inB);
    bitmask_t coords{0};

    while ((utB = inB / 2))
    {
        unsigned int const shiftAmt = nDims1 * utB;
        bitmask_t const utFieldEnds =
            inFieldEnds | (inFieldEnds << (shiftAmt + utB));
        bitmask_t const utMask =
            (utFieldEnds << utB) - utFieldEnds;
        bitmask_t utCoords{0};
        if (inB & 1)
        {
            bitmask_t const inFieldStarts = inFieldEnds << (inB - 1);
            unsigned int oddShift = 2 * shiftAmt;
            for (unsigned int d = 0; d < nDims; ++d)
            {
                bitmask_t in = inCoords & inMask;
                inCoords >>= inB;
                coords |= (in & inFieldStarts) << oddShift++;
                in &= ~inFieldStarts;
                in = (in | (in << shiftAmt)) & utMask;
                utCoords |= in << (d * utB);
            }
        }
        else
        {
            for (unsigned int d = 0; d < nDims; ++d)
            {
                bitmask_t in = inCoords & inMask;
                inCoords >>= inB;
                in = (in | (in << shiftAmt)) & utMask;
                utCoords |= in << (d * utB);
            }
        }
        inCoords = utCoords;
        inB = utB;
        inFieldEnds = utFieldEnds;
        inMask = utMask;
    }
    coords |= inCoords;
    return coords;
}

/*****************************************************************
 * hilbert_i2c
 *
 * Convert an index into a Hilbert curve to a set of coordinates.
 * Inputs:
 *  nDims:      Number of coordinate axes.
 *  nBits:      Number of bits per axis.
 *  index:      The index, contains nDims*nBits bits (so nDims*nBits must be <= 8*sizeof(bitmask_t)).
 * Outputs:
 *  coord:      The list of nDims coordinates, each with nBits bits.
 * Assumptions:
 *      nDims*nBits <= (sizeof index) * (bits_per_byte)
 */
void hilbert_i2c(unsigned int nDims, unsigned int nBits, bitmask_t index, bitmask_t coord[])
{
    if (nDims > 1)
    {
        bitmask_t coords;
        halfmask_t const nbOnes = ones(halfmask_t, nBits);
        if (nBits > 1)
        {
            unsigned int const nDimsBits = nDims * nBits;
            halfmask_t const ndOnes = ones(halfmask_t, nDims);
            halfmask_t const nd1Ones = ndOnes >> 1; /* for adjust_rotation */
            unsigned int b = nDimsBits;
            unsigned int rotation{0};
            halfmask_t flipBit{0};
            bitmask_t const nthbits = ones(bitmask_t, nDimsBits) / ndOnes;
            index ^= (index ^ nthbits) >> 1;
            coords = 0;
            do
            {
                halfmask_t bits = (index >> (b -= nDims)) & ndOnes;
                coords <<= nDims;
                coords |= rotateLeft(bits, rotation, nDims) ^ flipBit;
                flipBit = (halfmask_t)1 << rotation;
                adjust_rotation(rotation, nDims, bits);
            } while (b);
            for (b = nDims; b < nDimsBits; b *= 2)
                coords ^= coords >> b;
            coords = bitTranspose(nBits, nDims, coords);
        }
        else
            coords = index ^ (index >> 1);

        for (unsigned int d = 0; d < nDims; ++d)
        {
            coord[d] = coords & nbOnes;
            coords >>= nBits;
        }
    }
    else
        coord[0] = index;
}

/*****************************************************************
 * hilbert_c2i
 *
 * Convert coordinates of a point on a Hilbert curve to its index.
 * Inputs:
 *  nDims:      Number of coordinates.
 *  nBits:      Number of bits/coordinate.
 *  coord:      Array of n nBits-bit coordinates.
 * Outputs:
 *  index:      Output index value.  nDims*nBits bits.
 * Assumptions:
 *      nDims*nBits <= (sizeof bitmask_t) * (bits_per_byte)
 */
bitmask_t hilbert_c2i(unsigned int nDims, unsigned int nBits, bitmask_t const coord[])
{
    if (nDims > 1)
    {
        unsigned int const nDimsBits = nDims * nBits;
        bitmask_t coords{0};
        for (unsigned int d = nDims; d--;)
        {
            coords <<= nBits;
            coords |= coord[d];
        }

        bitmask_t index;
        if (nBits > 1)
        {
            halfmask_t const ndOnes = ones(halfmask_t, nDims);
            halfmask_t const nd1Ones = ndOnes >> 1; /* for adjust_rotation */
            unsigned int b = nDimsBits;
            unsigned int rotation{0};
            halfmask_t flipBit{0};
            bitmask_t const nthbits = ones(bitmask_t, nDimsBits) / ndOnes;
            coords = bitTranspose(nDims, nBits, coords);
            coords ^= coords >> nDims;
            index = 0;
            do
            {
                halfmask_t bits = (coords >> (b -= nDims)) & ndOnes;
                bits = rotateRight(flipBit ^ bits, rotation, nDims);
                index <<= nDims;
                index |= bits;
                flipBit = static_cast<halfmask_t>(1) << rotation;
                adjust_rotation(rotation, nDims, bits);
            } while (b);
            index ^= nthbits >> 1;
        }
        else
            index = coords;
        for (unsigned int d = 1; d < nDimsBits; d *= 2)
            index ^= index >> d;

        return index;
    }
    else
        return coord[0];
}

static bitmask_t getIntBits(unsigned int nDims, unsigned int nBytes, char const *c, unsigned int y)
{
    unsigned int const bit = y % 8;
    unsigned int const offs = whichByte(nBytes, y);
    c += offs;
    bitmask_t bits{0};
    for (unsigned int d = 0; d < nDims; ++d)
    {
        bits |= rdbit(*c, bit) << d;
        c += nBytes;
    }
    return bits;
}

static void propogateIntBits(unsigned int d, unsigned int nBytes, char *c, unsigned int y, int fold)
{
    unsigned int const byteId = whichByte(nBytes, y);
    unsigned int const b = y % 8;
    char const bthbit = 1 << b;
    char *const target = &c[d * nBytes];
    target[byteId] ^= bthbit;
    if (!fold)
    {
        char notbit = ((target[byteId] >> b) & 1) - 1;
        if (notbit)
            target[byteId] |= bthbit - 1;
        else
            target[byteId] &= -bthbit;
        setBytes(target, byteId, nBytes, notbit);
    }
}

typedef bitmask_t (*BitReader)(unsigned int nDims, unsigned int nBytes, char const *c, unsigned int y);
typedef void (*BitWriter)(unsigned int d, unsigned int nBytes, char *c, unsigned int y, int fold);

// An IEEE double is treated as a 2100 bit number.  In particular, 0 is treated
// as a 1 followed by 2099 zeroes, and negative 0 as a 0 followed by 2099 ones.
// Only 53 bits differ between a number and a zero of the same sign, with the
// position of the 53 determined by the exponent, and the values of the 53 by
// the significand (with implicit leading 1 bit).  Although IEEE 754 uses the
// maximum exponent for NaN's and infinities, this implementation ignores that
// decision, so that infinities and NaN's are treated as very large numbers.
// Note that we do not explicitly construct a 2100 bit bitmask in the IEEE
// routines below.
enum
{
    IEEEexpBits = 11
};
enum
{
    IEEEsigBits = 52
};
enum
{
    IEEErepBits = (1 << IEEEexpBits) + IEEEsigBits
};

union ieee754_double
{
    double d;

    /* This is the IEEE 754 double-precision format.  */
    struct
    {
#if defined(HILBERT_BIG_ENDIAN)
        unsigned int negative : 1;
        unsigned int exponent : 11;
        /* Together these comprise the mantissa.  */
        unsigned int mantissa0 : 20;
        unsigned int mantissa1 : 32;
#else  /* Big endian.  */
        /* Together these comprise the mantissa.  */
        unsigned int mantissa1 : 32;
        unsigned int mantissa0 : 20;
        unsigned int exponent : 11;
        unsigned int negative : 1;
#endif /* Little endian.  */
    } ieee;
};

static bitmask_t getIEEESignBits(unsigned int nDims, double const *c)
{
    bitmask_t bits{0};
    ieee754_double x;
    for (unsigned int d = 0; d < nDims; ++d)
    {
        x.d = c[d];
        bits |= x.ieee.negative << d;
    }
    return bits;
}

static bitmask_t getIEEEBits(unsigned int nDims, unsigned int /* ignoreMe */, char const *cP, unsigned int y)
// retrieve bits y of elements of double array c, where an expanded IEEE double has 2100 bits.
{
    unsigned int d;
    double const *c = (double const *)cP;
    bitmask_t bits{0};
    ieee754_double x;
    for (x.d = c[d = 0]; d < nDims; x.d = c[++d])
    {
        bitmask_t bit = x.ieee.negative;
        unsigned int normalized = (x.ieee.exponent != 0);
        unsigned int diff = y - (x.ieee.exponent - normalized);
        if (diff <= 52)
        {
            bit ^= 1 & ((diff < 32) ? x.ieee.mantissa1 >> diff : (diff < 52) ? x.ieee.mantissa0 >> (diff - 32)
                                                                             : normalized);
        }
        else
        {
            bit ^= (y == IEEErepBits - 1);
        }
        bits |= bit << d;
    }

    return bits;
}

static void propogateIEEEBits(unsigned int d, unsigned int nBytes, char *cP, unsigned int y, int fold)
{
    ieee754_double *x = d + (ieee754_double *)cP;
    unsigned int normalized = (x->ieee.exponent != 0);
    unsigned int diff = y - (x->ieee.exponent - normalized);
    if (diff < 32)
    {
        unsigned int b = 1 << diff;
        unsigned int bit = x->ieee.mantissa1 & b;
        x->ieee.mantissa1 &= ~(b - 1);
        x->ieee.mantissa1 |= b;
        if (bit)
            --x->ieee.mantissa1;
    }
    else if (diff < 52)
    {
        unsigned int b = 1 << (diff - 32);
        unsigned int bit = x->ieee.mantissa0 & b;
        x->ieee.mantissa0 &= ~(b - 1);
        x->ieee.mantissa0 |= b;
        if (bit)
            --x->ieee.mantissa0;
        x->ieee.mantissa1 = bit ? -1 : 0;
    }
    else if (diff == 52) /* "flip" the implicit 1 bit */
    {
        if (normalized)
            --x->ieee.exponent;
        else
            x->ieee.exponent = 1;
        x->ieee.mantissa0 = -normalized;
        x->ieee.mantissa1 = -normalized;
    }
    else if (diff < IEEErepBits)
    {
        if (y == IEEErepBits - 1)
        {
            x->ieee.negative ^= 1;
            x->ieee.exponent = 0;
        }
        else
            x->ieee.exponent = y - 51;
        x->ieee.mantissa0 = 0;
        x->ieee.mantissa1 = 0;
    }
}

static unsigned int getIEEEexptMax(unsigned int nDims, double const *c)
{
    unsigned int max{0};
    for (unsigned int d = 0; d < nDims; ++d)
    {
        ieee754_double x;
        x.d = c[d];
        if (max < x.ieee.exponent)
        {
            max = x.ieee.exponent;
        }
    }
    if (max)
    {
        --max;
    }
    return max;
}

static void getIEEEinitValues(double const *c1,
                              unsigned int y,
                              unsigned int nDims,
                              unsigned int *rotation,
                              bitmask_t *bits,
                              bitmask_t *index)
{
    bitmask_t signBits = getIEEESignBits(nDims, c1);

    // compute the odd/evenness of the number of sign bits
    unsigned int signParity;
    {
        bitmask_t signPar = signBits;
        for (unsigned int d = 1; d < nDims; d *= 2)
        {
            signPar ^= signPar >> d;
        }
        signParity = signPar & 1;
    }

    // find the position of the least-order 0 bit in among signBits and adjust it if necessary
    unsigned int leastZeroBit;
    for (leastZeroBit = 0; leastZeroBit < nDims; ++leastZeroBit)
    {
        if (rdbit(signBits, leastZeroBit) == 0)
        {
            break;
        }
    }

    unsigned int strayBit = (leastZeroBit == nDims - 2) ? 1 : (leastZeroBit == nDims) ? nDims - 1
                                                                                      : 0;

    if (y % 2 == 1)
    {
        *rotation = (IEEErepBits - y + 1 + leastZeroBit) % nDims;
        if (y < IEEErepBits - 1)
        {
            *bits = signBits ^ (static_cast<bitmask_t>(1) << ((*rotation + strayBit) % nDims));
            *index = signParity;
        }
        else /* y == IEEErepBits-1 */
        {
            *bits = signBits ^ (ones(bitmask_t, nDims) & ~1);
            *index = signParity ^ (nDims & 1);
        }
    }
    else /* y % 2 == 0 */
    {
        if (y < IEEErepBits)
        {
            unsigned int shift_amt = (IEEErepBits - y + leastZeroBit) % nDims;
            *rotation = (shift_amt + 2 + strayBit) % nDims;
            *bits = signBits ^ (static_cast<bitmask_t>(1) << shift_amt);
            *index = signParity ^ 1;
        }
        else /* y == IEEErepBits */
        {
            *rotation = 0;
            *bits = static_cast<bitmask_t>(1) << (nDims - 1);
            *index = 1;
        }
    }
}

static int hilbert_cmp_work(unsigned int nDims, unsigned int nBytes, unsigned int nBits,
                            unsigned int max, unsigned int y,
                            char const *c1, char const *c2,
                            unsigned int rotation,
                            bitmask_t bits,
                            bitmask_t index,
                            BitReader getBits)
{
    bitmask_t const nd1Ones = ones(bitmask_t, nDims) >> 1; /* used in adjust_rotation macro */
    while (y-- > max)
    {
        bitmask_t reflection = getBits(nDims, nBytes, c1, y);
        bitmask_t diff = reflection ^ getBits(nDims, nBytes, c2, y);
        bits ^= reflection;
        bits = rotateRight(bits, rotation, nDims);
        if (diff)
        {
            diff = rotateRight(diff, rotation, nDims);
            for (unsigned int d = 1; d < nDims; d *= 2)
            {
                index ^= index >> d;
                bits ^= bits >> d;
                diff ^= diff >> d;
            }
            return (((index ^ y ^ nBits) & 1) == (bits < (bits ^ diff))) ? -1 : 1;
        }
        index ^= bits;
        reflection ^= static_cast<bitmask_t>(1) << rotation;
        adjust_rotation(rotation, nDims, bits);
        bits = reflection;
    }
    return 0;
}

/*****************************************************************
 * hilbert_cmp, hilbert_ieee_cmp
 *
 * Determine which of two points lies further along the Hilbert curve
 * Inputs:
 *  nDims:      Number of coordinates.
 *  nBytes:     Number of bytes of storage/coordinate (hilbert_cmp only)
 *  nBits:      Number of bits/coordinate. (hilbert_cmp only)
 *  coord1:     Array of nDims nBytes-byte coordinates (or doubles for ieee_cmp).
 *  coord2:     Array of nDims nBytes-byte coordinates (or doubles for ieee_cmp).
 * Return value:
 *      -1, 0, or 1 according to whether
           coord1<coord2, coord1==coord2, coord1>coord2
 * Assumptions:
 *      nBits <= (sizeof bitmask_t) * (bits_per_byte)
 */
int hilbert_cmp(unsigned int nDims, unsigned int nBytes, unsigned int nBits, void const *coord1, void const *coord2)
{
    bitmask_t bits = static_cast<bitmask_t>(1) << (nDims - 1);
    return hilbert_cmp_work(nDims, nBytes, nBits, 0, nBits, (char const *)coord1, (char const *)coord2, 0, bits, bits, getIntBits);
}

int hilbert_ieee_cmp(unsigned int nDims, double const *coord1, double const *coord2)
{
    unsigned int rotation, max;
    bitmask_t bits, index;
    if (getIEEESignBits(nDims, coord1) != getIEEESignBits(nDims, coord2))
    {
        max = 2047;
    }
    else
    {
        unsigned int max1 = getIEEEexptMax(nDims, coord1);
        unsigned int max2 = getIEEEexptMax(nDims, coord2);
        max = (max1 > max2) ? max1 : max2;
    }

    getIEEEinitValues(coord1, max + 53, nDims, &rotation, &bits, &index);
    return hilbert_cmp_work(nDims, 8, 64, max, max + 53, (char const *)coord1, (char const *)coord2, rotation, bits, index, getIEEEBits);
}

static unsigned int hilbert_box_vtx_work(unsigned int nDims,
                                         unsigned int nBytes,
                                         unsigned int nBits,
                                         int findMin,
                                         unsigned int max,
                                         unsigned int y,
                                         char *c1,
                                         char *c2,
                                         unsigned int rotation,
                                         bitmask_t bits,
                                         bitmask_t index,
                                         BitReader getBits)
{
    bitmask_t const ndOnes = ones(bitmask_t, nDims);
    bitmask_t const nd1Ones = ndOnes >> 1;
    bitmask_t bitsFolded{0};

    while (y--)
    {
        bitmask_t reflection = getBits(nDims, nBytes, c1, y);
        bitmask_t diff = reflection ^ getBits(nDims, nBytes, c2, y);
        if (diff)
        {
            bitmask_t smear = rotateRight(diff, rotation, nDims) >> 1;
            bitmask_t digit = rotateRight(bits ^ reflection, rotation, nDims);
            for (unsigned int d = 1; d < nDims; d *= 2)
            {
                index ^= index >> d;
                digit ^= (digit >> d) & ~smear;
                smear |= smear >> d;
            }
            index &= 1;
            if ((index ^ y ^ findMin) & 1)
                digit ^= smear + 1;
            digit = rotateLeft(digit, rotation, nDims) & diff;
            reflection ^= digit;

            for (unsigned int d = 0; d < nDims; ++d)
                if (rdbit(diff, d))
                {
                    int way = rdbit(digit, d);
                    char *target = d * nBytes + (way ? c1 : c2);
                    char *const source = 2 * d * nBytes + c1 - target + c2;
                    std::memcpy(target, source, nBytes);
                }

            bitsFolded |= diff;
            if (bitsFolded == ndOnes)
                return y;
        }

        bits ^= reflection;
        bits = rotateRight(bits, rotation, nDims);
        index ^= bits;
        reflection ^= static_cast<bitmask_t>(1) << rotation;
        adjust_rotation(rotation, nDims, bits);
        bits = reflection;
    }
    return y;
}

/*****************************************************************
 * hilbert_box_vtx
 *
 * Determine the first or last vertex of a box to lie on a Hilbert curve
 * Inputs:
 *  nDims:      Number of coordinates.
 *  nBytes:     Number of bytes/coordinate.
 *  nBits:      Number of bits/coordinate. (hilbert_cmp only)
 *  findMin:    Is it the least vertex sought?
 *  coord1:     Array of nDims nBytes-byte coordinates - one corner of box
 *  coord2:     Array of nDims nBytes-byte coordinates - opposite corner
 * Output:
 *      c1 and c2 modified to refer to selected corner
 *      value returned is log2 of size of largest power-of-two-aligned box that
 *      contains the selected corner and no other corners
 * Assumptions:
 *      nBits <= (sizeof bitmask_t) * (bits_per_byte)
 */
unsigned int hilbert_box_vtx(unsigned int nDims, unsigned int nBytes, unsigned int nBits,
                             int findMin, void *c1, void *c2)
{
    bitmask_t bits = static_cast<bitmask_t>(1) << (nDims - 1);
    return hilbert_box_vtx_work(nDims, nBytes, nBits, findMin, 0, nBits, (char *)c1, (char *)c2, 0, bits, bits, getIntBits);
}

unsigned int hilbert_ieee_box_vtx(unsigned int nDims, int findMin, double *c1, double *c2)
{
    unsigned int rotation, max;
    bitmask_t bits, index;
    if (getIEEESignBits(nDims, c1) != getIEEESignBits(nDims, c2))
    {
        max = 2047;
    }
    else
    {
        unsigned int max1 = getIEEEexptMax(nDims, c1);
        unsigned int max2 = getIEEEexptMax(nDims, c2);
        max = (max1 > max2) ? max1 : max2;
    }

    getIEEEinitValues(c1, max + 53, nDims, &rotation, &bits, &index);

    return hilbert_box_vtx_work(nDims, 8, 64, findMin, max, max + 53, (char *)c1, (char *)c2, rotation, bits, index, getIEEEBits);
}

// hilbert_box_pt
//
// Determine the first or last point of a box to lie on a Hilbert curve
// Inputs:
//  nDims:      Number of coordinates.
//  nBytes:     Number of bytes/coordinate.
//  nBits:      Number of bits/coordinate.
//  findMin:    Is it the least vertex sought?
//  coord1:     Array of nDims nBytes-byte coordinates - one corner of box
//  coord2:     Array of nDims nBytes-byte coordinates - opposite corner
// Output:
//      c1 and c2 modified to refer to least point
// Assumptions:
//      nBits <= (sizeof bitmask_t) * (bits_per_byte)
unsigned int hilbert_box_pt_work(unsigned int nDims,
                                 unsigned int nBytes,
                                 unsigned int nBits,
                                 int findMin,
                                 unsigned int max,
                                 unsigned int y,
                                 char *c1,
                                 char *c2,
                                 unsigned int rotation,
                                 bitmask_t bits,
                                 bitmask_t index,
                                 BitReader getBits,
                                 BitWriter propogateBits)
{
    bitmask_t const one{1};
    bitmask_t const nd1Ones = ones(bitmask_t, nDims) >> 1;
    bitmask_t fold1{0};
    bitmask_t fold2{0};
    unsigned int smearSum{0};

    while (y-- > max)
    {
        bitmask_t reflection = getBits(nDims, nBytes, c1, y);
        bitmask_t diff = reflection ^ getBits(nDims, nBytes, c2, y);
        if (diff)
        {
            bitmask_t smear = rotateRight(diff, rotation, nDims) >> 1;
            bitmask_t digit = rotateRight(bits ^ reflection, rotation, nDims);
            for (unsigned int d = 1; d < nDims; d *= 2)
            {
                index ^= index >> d;
                digit ^= (digit >> d) & ~smear;
                smear |= smear >> d;
            }
            smearSum += smear;
            index &= 1;
            if ((index ^ y ^ findMin) & 1)
                digit ^= smear + 1;
            digit = rotateLeft(digit, rotation, nDims) & diff;
            reflection ^= digit;

            for (unsigned int d = 0; d < nDims; ++d)
                if (rdbit(diff, d))
                {
                    int way = rdbit(digit, d);
                    char *c = way ? c1 : c2;
                    bitmask_t fold = way ? fold1 : fold2;
                    propogateBits(d, nBytes, c, y, rdbit(fold, d));
                }
            diff ^= digit;
            fold1 |= digit;
            fold2 |= diff;
        }

        bits ^= reflection;
        bits = rotateRight(bits, rotation, nDims);
        index ^= bits;
        reflection ^= one << rotation;
        adjust_rotation(rotation, nDims, bits);
        bits = reflection;
    }
    return smearSum;
}

/*****************************************************************
 * hilbert_box_pt
 *
 * Determine the first or last point of a box to lie on a Hilbert curve
 * Inputs:
 *  nDims:      Number of coordinates.
 *  nBytes:     Number of bytes/coordinate.
 *  nBits:      Number of bits/coordinate.
 *  findMin:    Is it the least vertex sought?
 *  coord1:     Array of nDims nBytes-byte coordinates - one corner of box
 *  coord2:     Array of nDims nBytes-byte coordinates - opposite corner
 * Output:
 *      c1 and c2 modified to refer to least point
 * Assumptions:
 *      nBits <= (sizeof bitmask_t) * (bits_per_byte)
 */
unsigned int hilbert_box_pt(unsigned int nDims, unsigned int nBytes, unsigned int nBits,
                            int findMin, void *coord1, void *coord2)
{
    bitmask_t bits = static_cast<bitmask_t>(1) << (nDims - 1);
    return hilbert_box_pt_work(nDims, nBytes, nBits, findMin, 0, nBits, (char *)coord1, (char *)coord2, 0, bits, bits, getIntBits, propogateIntBits);
}

unsigned int hilbert_ieee_box_pt(unsigned int nDims, int findMin, double *c1, double *c2)
{
    unsigned int rotation, max;
    bitmask_t bits, index;
    bitmask_t c1Signs = getIEEESignBits(nDims, c1);
    bitmask_t c2Signs = getIEEESignBits(nDims, c2);
    if (c1Signs != c2Signs)
    {
        rotation = 0;
        bits = (bitmask_t)1 << (nDims - 1);
        index = 1;
        hilbert_box_pt_work(nDims, 8, 64, findMin, IEEErepBits - 1, IEEErepBits, (char *)c1, (char *)c2,
                            rotation, bits, index, getIEEEBits, propogateIEEEBits);
    }

    /* having put everything in the same orthant, start */
    {
        unsigned int max1 = getIEEEexptMax(nDims, c1);
        unsigned int max2 = getIEEEexptMax(nDims, c2);
        max = (max1 > max2) ? max1 : max2;
    }

    getIEEEinitValues(c1, max + 53, nDims, &rotation, &bits, &index);

    return hilbert_box_pt_work(nDims, 8, 64, findMin, max, max + 53, (char *)c1, (char *)c2,
                               rotation, bits, index, getIEEEBits, propogateIEEEBits);
}

/*****************************************************************
 * hilbert_nextinbox
 *
 * Determine the first point of a box after a given point to lie on a Hilbert curve
 * Inputs:
 *  nDims:      Number of coordinates.
 *  nBytes:     Number of bytes/coordinate.
 *  nBits:      Number of bits/coordinate.
 *  findPrev:   Is the previous point sought?
 *  coord1:     Array of nDims nBytes-byte coordinates - one corner of box
 *  coord2:     Array of nDims nBytes-byte coordinates - opposite corner
 *  point:      Array of nDims nBytes-byte coordinates - lower bound on point returned
 *
 * Output:
      if returns 1:
 *      coord1 and coord2 modified to refer to least point after "point" in box
      else returns 0:
        arguments unchanged; "point" is beyond the last point of the box
 * Assumptions:
 *      nBits <= (sizeof bitmask_t) * (bits_per_byte)
 */
int hilbert_nextinbox(unsigned int nDims,
                      unsigned int nBytes,
                      unsigned int nBits,
                      int findPrev,
                      void *coord1,
                      void *coord2,
                      void const *point)
{
    unsigned int y{nBits};
    bitmask_t const ndOnes = ones(bitmask_t, nDims);
    bitmask_t const nd1Ones = ndOnes >> 1;
    unsigned int rotation{0};
    bitmask_t bits{0};
    bitmask_t index{0};
    bitmask_t fold1{0};
    bitmask_t fold2{0};
    bitmask_t valu1{0};
    bitmask_t valu2{0};
    unsigned int p_y;
    bitmask_t p_separator{0};
    bitmask_t p_firstSeparator;
    bitmask_t p_cornerdiff, p_reflection;
    bitmask_t p_fold1, p_fold2, p_valu1, p_valu2;

    char *c1 = (char *)coord1;
    char *c2 = (char *)coord2;
    char const *pt = (const char *)point;

    while (y-- > 0)
    {
        bitmask_t reflection = getIntBits(nDims, nBytes, pt, y);
        bitmask_t diff = reflection ^ /* planes that separate box and point */
                         ((getIntBits(nDims, nBytes, c1, y) & ~fold1) | valu1);

        if (diff)
        /* some coordinate planes separate point from box or
           dividing box or both; smear the bits of diff to reflect that
           after the first diff dimension, they might as well all be
           diffing; adjust the diff to reflect the fact that diffed
           dimensions don't matter. */
        {
            /* compute (the complement of) a "digit" in the integer index of this
               point */
            bitmask_t cornerdiff = (diff ^ reflection) ^ /* separate box crnrs */
                                   ((getIntBits(nDims, nBytes, c2, y) & ~fold2) | valu2);
            bitmask_t separator = diff & ~cornerdiff;
            /* eventually, the most significant separating cutting plane */
            bitmask_t firstSeparator;
            /* bits less significant than the msb of separator are irrelevant;
               for convenience, call them all separators too */
            bitmask_t rotSep = rotateRight(separator, rotation, nDims);
            /* compute the (complement of the) digit of the hilbert code
               assoc with point */
            bitmask_t digit = rotateRight(bits ^ reflection, rotation, nDims);
            for (unsigned int d = 1; d < nDims; d *= 2)
            {
                index ^= index >> d;
                digit ^= digit >> d;
                rotSep |= rotSep >> d;
            }
            index &= 1;
            digit &= rotSep;
            if ((index ^ y ^ findPrev) & 1)
                digit ^= rotSep;

            separator = rotateLeft(rotSep, rotation, nDims);
            rotSep -= rotSep >> 1;
            firstSeparator = rotateLeft(rotSep, rotation, nDims);
            /* forget about all the planes that split the box, except those that
               are more significant than the most significant separator. */
            cornerdiff &= ~separator;

            if (cornerdiff && digit)
            /* some coordinate planes divide the box.  Call the part of the
               box in the same orthant as the point "here" and the part of
               the box in the next (or previous) orthant "there".  Remember
               what the "there" orthant of the box looks like in case it
               turns out that the curve doesn't reenter the box "here" after
               (before) passing thru point.  Continue working with the
               "here" part. If there is no "there" there, skip it */
            {
                p_firstSeparator = digit & -digit;
                p_separator = 2 * p_firstSeparator - 1;
                p_separator = rotateLeft(p_separator, rotation, nDims);
                p_firstSeparator = rotateLeft(p_firstSeparator, rotation, nDims);
                p_cornerdiff = cornerdiff & ~(p_separator ^ p_firstSeparator);
                p_y = y;
                p_reflection = reflection ^ p_firstSeparator;
                p_fold1 = fold1;
                p_fold2 = fold2;
                p_valu1 = valu1;
                p_valu2 = valu2;
            }

            if (digit < rotSep)

            /* use next box */
            {
                if (!p_separator)
                    return 0; /* no next point */
                separator = p_separator;
                firstSeparator = p_firstSeparator;
                y = p_y;
                cornerdiff = p_cornerdiff;
                reflection = p_reflection;
                fold1 = p_fold1;
                fold2 = p_fold2;
                valu1 = p_valu1;
                valu2 = p_valu2;
            }

            if (cornerdiff)
            {
                /* reduce currbox */
                bitmask_t corner = diff & cornerdiff;
                cornerdiff ^= corner;
                fold1 |= corner;
                fold2 |= cornerdiff;
                valu1 |= ~reflection & corner;
                valu2 |= ~reflection & cornerdiff;
            }

            separator ^= firstSeparator;
            if (firstSeparator)
            /* we have completely separated the point from a part of the box
               ahead of it on the curve; almost done */
            {
                unsigned int byteId = whichByte(nBytes, y);
                bitmask_t bthbit = static_cast<bitmask_t>(1) << y % 8;
                for (unsigned int d = 0; d < nDims; ++d)
                {
                    char lo1, lo2;
                    char *cc1 = &c1[d * nBytes];
                    char *cc2 = &c2[d * nBytes];
                    char const *pnt = &pt[d * nBytes];
                    char hibits = -bthbit;
                    char hipart = pnt[byteId] & hibits;
                    std::memcpy(cc1, pnt, byteId);
                    std::memcpy(cc2, pnt, byteId);

                    if (rdbit(separator, d))
                        hibits ^= bthbit;
                    if (rdbit(firstSeparator, d))
                        hipart ^= bthbit;

                    if (rdbit(fold1, d))
                    {
                        lo1 = -rdbit(valu1, d);
                        setBytes(cc1, byteId, nBytes, lo1);
                    }
                    else
                        lo1 = cc1[byteId];
                    cc1[byteId] = hipart | (lo1 & ~hibits);

                    if (rdbit(fold2, d))
                    {
                        lo2 = -rdbit(valu2, d);
                        setBytes(cc2, byteId, nBytes, lo2);
                    }
                    else
                        lo2 = cc2[byteId];
                    cc2[byteId] = hipart | (lo2 & ~hibits);
                }

                hilbert_box_pt(nDims, nBytes, nBits, !findPrev, coord1, coord2);
                return 1;
            }
        }

        bits ^= reflection;
        bits = rotateRight(bits, rotation, nDims);
        index ^= bits;
        reflection ^= static_cast<bitmask_t>(1) << rotation;
        adjust_rotation(rotation, nDims, bits);
        bits = reflection;
    }

    /* point is in box */
    {
        for (unsigned int d = 0; d < nDims; ++d)
        {
            ((char *)c1)[d] = ((char *)c2)[d] = ((char *)pt)[d];
        }
    }
    return 1;
}

/*****************************************************************
 * hilbert_incr
 *
 * Advance from one point to its successor on a Hilbert curve
 * Inputs:
 *  nDims:      Number of coordinates.
 *  nBits:      Number of bits/coordinate.
 *  coord:      Array of nDims nBits-bit coordinates.
 * Output:
 *  coord:      Next point on Hilbert curve
 * Assumptions:
 *      nBits <= (sizeof bitmask_t) * (bits_per_byte)
 */
void hilbert_incr(unsigned int nDims, unsigned int nBits, bitmask_t coord[])
{
    bitmask_t const ndOnes = ones(bitmask_t, nDims);
    bitmask_t const nd1Ones = ndOnes >> 1;
    unsigned int rotation{0};
    bitmask_t reflection{0};
    bitmask_t index{0};
    unsigned int rb{nBits - 1};
    bitmask_t rd{ndOnes};
    unsigned int d;
    for (unsigned int b = nBits; b--;)
    {
        bitmask_t bits = reflection;
        reflection = 0;
        for (d = 0; d < nDims; ++d)
        {
            reflection |= rdbit(coord[d], b) << d;
        }
        bits ^= reflection;
        bits = rotateRight(bits, rotation, nDims);
        index ^= bits;
        for (d = 1; d < nDims; d *= 2)
        {
            index ^= index >> d;
        }
        if (index++ != ndOnes)
        {
            rb = b;
            rd = index & -index;
            rd = rotateLeft(rd, rotation, nDims);
        }
        index &= 1;
        index <<= nDims - 1;

        reflection ^= static_cast<bitmask_t>(1) << rotation;
        adjust_rotation(rotation, nDims, bits);
    }
    for (d = 0; !rdbit(rd, d); ++d)
    {
    }
    coord[d] ^= (2 << rb) - 1;
}

#endif // HILBERT_HPP
