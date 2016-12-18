{-# LANGUAGE GADTs, RankNTypes, MultiParamTypeClasses, FlexibleInstances, UndecidableInstances, KindSignatures, ScopedTypeVariables, ConstraintKinds, TemplateHaskell, GeneralizedNewtypeDeriving #-}

module Numeric.Trainee.Types (
	Gradee(..),
	Ow(..), NoParams(..), PairParams(..),
	Parametric, LearneeT(..), params, forwardPass, Learnee(..), toLearnee, withLearnee, overLearnee, onLearnee,
	Cost,
	Sample
	) where

import Prelude hiding (id, (.))
import Prelude.Unicode

import Control.Category
import Control.DeepSeq
import Control.Lens
import Data.List (intersperse, intercalate)

data Gradee a b = Gradee {
	runGradee ∷ Lens' a b }

instance Category Gradee where
	id = Gradee id
	Gradee f . Gradee g = Gradee (g . f)

class Category a ⇒ Ow a where
	first ∷ a b c → a (b, d) (c, d)
	first = flip stars id

	second ∷ a b c → a (d, b) (d, c)
	second = stars id

	stars ∷ a b c → a b' c' → a (b, b') (c, c')

instance Ow Gradee where
	stars (Gradee f) (Gradee g) = Gradee $ lens g' s' where
		g' (x, y) = (view f x, view g y)
		s' (x, y) (x', y') = (set f x' x, set g y' y)

data NoParams = NoParams deriving (Eq, Ord, Read, Enum, Bounded)

instance NFData NoParams where
	rnf NoParams = ()

instance Show NoParams where
	show NoParams = ""

instance Num NoParams where
	_ + _ = NoParams
	_ * _ = NoParams
	abs _ = NoParams
	signum _ = NoParams
	fromInteger _ = NoParams
	negate _ = NoParams

instance Fractional NoParams where
	fromRational _ = NoParams
	recip _ = NoParams

data PairParams l r = PairParams l r deriving (Eq, Ord, Read)

instance (NFData l, NFData r) ⇒ NFData (PairParams l r) where
	rnf (PairParams x y) = rnf x `seq` rnf y

instance (Show l, Show r) ⇒ Show (PairParams l r) where
	show (PairParams x y) = intercalate "\n" $ intersperse (replicate 10 '-') $
		filter (not ∘ null) [show x, show y]

instance (Num l, Num r) ⇒ Num (PairParams l r) where
	PairParams l r + PairParams l' r' = PairParams (l + l') (r + r')
	PairParams l r * PairParams l' r' = PairParams (l * l') (r * r')
	abs (PairParams l r) = PairParams (abs l) (abs r)
	signum (PairParams l r) = PairParams (signum l) (signum r)
	fromInteger i = PairParams (fromInteger i) (fromInteger i)
	negate (PairParams l r) = PairParams (negate l) (negate r)

instance (Fractional l, Fractional r) ⇒ Fractional (PairParams l r) where
	fromRational r = PairParams (fromRational r) (fromRational r)
	recip (PairParams l r) = PairParams (recip l) (recip r)

type Parametric w = (Read w, Show w, Num w, Fractional w, NFData w)

data LearneeT w a b = LearneeT {
	_params ∷ w,
	_forwardPass ∷ w → a → (b, b → (a, w)) }

makeLenses ''LearneeT

instance NFData w ⇒ NFData (LearneeT w a b) where
	rnf (LearneeT ws _) = rnf ws

instance Show w ⇒ Show (LearneeT w a b) where
	show (LearneeT ws _) = show ws

data Learnee a b where
	Learnee ∷ Parametric w ⇒ w → (w → a → (b, b → (a, w))) → Learnee a b

toLearnee ∷ Parametric w ⇒ LearneeT w a b → Learnee a b
toLearnee (LearneeT ws fn) = Learnee ws fn

instance NFData (Learnee a b) where
	rnf (Learnee ws _) = rnf ws

instance Show (Learnee a b) where
	show (Learnee ws _) = show ws

withLearnee ∷ Applicative f ⇒ (forall w . Parametric w ⇒ LearneeT w a b → f (LearneeT w a b)) → Learnee a b → f (Learnee a b)
withLearnee act (Learnee ws fn) = toLearnee <$> act (LearneeT ws fn)

overLearnee ∷ (forall w . Parametric w ⇒ LearneeT w a b → LearneeT w a b) → Learnee a b → Learnee a b
overLearnee fn = runIdentity ∘ withLearnee (pure ∘ fn)

onLearnee ∷ (forall w . Parametric w ⇒ LearneeT w a b → r) → Learnee a b → r
onLearnee fn = fromLeft ∘ withLearnee (Left ∘ fn) where
	fromLeft (Left v) = v
	fromLeft _ = error "onLearnee"

type Cost b = b → b → (b, b)

type Sample a b = (a, b)
