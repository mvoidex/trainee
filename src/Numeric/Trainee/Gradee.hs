{-# LANGUAGE RankNTypes, FlexibleContexts, TypeFamilies #-}

module Numeric.Trainee.Gradee (
	Gradee(..), gradee, ad,
	Unary, Binary,
	unary, binary,
	dup, conjoin, swap,

	matMat, matVec, odot
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

-- | Make Gradee from any function
ad ∷ (Traversable f, Num a) ⇒ (forall s . Reifies s Tape ⇒ f (Reverse s a) → Reverse s a) → Gradee (f a) a
ad f = gradee f' (\x _ → grad f x) where
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
matMat = gradee (uncurry (<>)) backprop where
	backprop (a, b) dc = (dc <> tr b, tr a <> dc)

matVec ∷ Numeric a ⇒ Gradee (Matrix a, Vector a) (Vector a)
matVec = gradee (uncurry (#>)) backprop where
	backprop (a, b) dc = (outer dc b, tr a #> dc)

odot ∷ Num (Vector a) ⇒ Gradee (Vector a, Vector a) (Vector a)
odot = gradee (uncurry (+)) backprop where
	backprop _ dc = (dc, dc)
