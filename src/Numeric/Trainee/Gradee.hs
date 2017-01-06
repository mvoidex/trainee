{-# LANGUAGE RankNTypes, FlexibleContexts, TypeFamilies #-}

module Numeric.Trainee.Gradee (
	Gradee(..), gradee, ad,
	Unary, Binary,
	unary, binary,
	dup, conjoin, swap,

	matMat, matVec, odot,
	corrVec, corrMat,
	flattenMat, reshapeVec,
	transposeMat,
	vecRow, vecCol,
	biasVec, biasMat
	) where

import Prelude hiding (id, (.))
import Prelude.Unicode

import Control.Category
import Control.Lens (lens)
import Data.Reflection (Reifies, reify)
import Numeric.AD (grad, auto)
import Numeric.AD.Internal.Reverse (Reverse, primal, Tape)
import Numeric.LinearAlgebra

import Numeric.Trainee.Types

-- | Make Gradee like lens
gradee ∷ (a → b) → (a → b → a) → Gradee a b
gradee g s = Gradee $ lens g s

-- | Make like lens from binary op
gradee2 ∷ (a → b → c) → (a → b → c → (a, b)) → Gradee (a, b) c
gradee2 g s = gradee (uncurry g) (uncurry s)

-- | Make Gradee from any function
ad ∷ (Traversable f, Num a) ⇒ (forall s . Reifies s Tape ⇒ f (Reverse s a) → Reverse s a) → Gradee (f a) a
ad f = gradee f' (\x dx → fmap (* dx) (grad f x)) where
	f' = reify undefined (\p → primal ∘ spec p f ∘ fmap auto)
	spec ∷ Reifies t Tape ⇒ proxy t → (forall s . Reifies s Tape ⇒ g (Reverse s a) → Reverse s a) → g (Reverse t a) → Reverse t a
	spec _ h = h

type Unary a = forall s . Reifies s Tape ⇒ Reverse s a → Reverse s a
type Binary a = forall s . Reifies s Tape ⇒ Reverse s a → Reverse s a → Reverse s a

-- | Make Gradee from unary function
unary ∷ Num a ⇒ Unary a → Gradee a a
unary f = ad (\[x] → f x) . gradee return (const head)

-- | Make @Gradee@ from binary function
binary ∷ Num a ⇒ Binary a → Gradee (a, a) a
binary f = ad (\[x, y] → f x y) . gradee g s where
	g (x, y) = [x, y]
	s _ [x, y] = (x, y)
	s _ _ = error "binary"

dup ∷ Num a ⇒ Gradee a (a, a)
dup = gradee (\x → (x, x)) (\_ (dx', dx'') → dx' + dx'')

conjoin ∷ Num a ⇒ Gradee (a, a) a
conjoin = binary (+)

swap ∷ Gradee (a, b) (b, a)
swap = gradee (\(x, y) → (y, x)) (\_ (dy, dx) → (dx, dy))

matMat ∷ Numeric a ⇒ Gradee (Matrix a, Matrix a) (Matrix a)
matMat = gradee2 (<>) backprop where
	backprop a b dc = (dc <> tr b, tr a <> dc)

matVec ∷ Numeric a ⇒ Gradee (Matrix a, Vector a) (Vector a)
matVec = gradee2 (#>) backprop where
	backprop a b dc = (outer dc b, tr a #> dc)

odot ∷ Num (Vector a) ⇒ Gradee (Vector a, Vector a) (Vector a)
odot = gradee2 (+) backprop where
	backprop _ _ dc = (dc, dc)

corrVec ∷ Numeric a ⇒ Gradee (Vector a, Vector a) (Vector a)
corrVec = gradee2 corr backprop where
	backprop a b dc = (corr dc b, conv a dc)

corrMat ∷ (Numeric a, Num (Vector a)) ⇒ Gradee (Matrix a, Matrix a) (Matrix a)
corrMat = gradee2 corr2 backprop where
	backprop a b dc = (corr2 dc b, conv2 a dc)

flattenMat ∷ Numeric a ⇒ Gradee (Matrix a) (Vector a)
flattenMat = gradee flatten backprop where
	backprop a = reshape (cols a)

reshapeVec ∷ Numeric a ⇒ Int → Gradee (Vector a) (Matrix a)
reshapeVec cols' = gradee (reshape cols') (const flatten)

transposeMat ∷ Numeric a ⇒ Gradee (Matrix a) (Matrix a)
transposeMat = gradee tr (const tr)

vecRow ∷ Numeric a ⇒ Gradee (Vector a) (Matrix a)
vecRow = gradee asRow (const flatten)

vecCol ∷ Numeric a ⇒ Gradee (Vector a) (Matrix a)
vecCol = gradee asColumn (const flatten)

biasVec ∷ Numeric a ⇒ Gradee (a, Vector a) (Vector a)
biasVec = gradee2 (\b v → cmap (+ b) v) backprop where
	backprop _ _ dv = (sumElements dv, dv)

biasMat ∷ Numeric a ⇒ Gradee (a, Matrix a) (Matrix a)
biasMat = gradee2 (\b m → cmap (+ b) m) backprop where
	backprop _ _ dm = (sumElements dm, dm)
