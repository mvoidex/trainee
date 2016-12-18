{-# LANGUAGE GADTs, RankNTypes, MultiParamTypeClasses, FlexibleInstances, UndecidableInstances, KindSignatures, ScopedTypeVariables, ConstraintKinds, TemplateHaskell, GeneralizedNewtypeDeriving #-}

module Numeric.Trainee.Types (
	Gradee(..),
	Ow(..), NoParams(..), Chain(..), Named(..),
	Parametric, LearneeT(..), params, pass, mapP, name, Learnee(..), withLearnee, withParams, toLearnee,
	LookupParams(..),
	Cost,
	Example
	) where

import Prelude hiding (id, (.))
import Prelude.Unicode

import Control.Applicative
import qualified Control.Arrow as A (second)
import Control.Category
import Control.Lens

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

data NoParams = NoParams deriving (Eq, Ord, Read, Show, Enum, Bounded)

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

newtype Chain l r = Chain { getChain ∷ (l, r) } deriving (Eq, Ord, Read)

instance (Show l, Show r) ⇒ Show (Chain l r) where
	show (Chain (x, y)) = unlines [show x, " -- chain -- ", show y]

instance (Num l, Num r) ⇒ Num (Chain l r) where
	Chain (l, r) + Chain (l', r') = Chain (l + l', r + r')
	Chain (l, r) * Chain (l', r') = Chain (l * l', r * r')
	abs (Chain (l, r)) = Chain (abs l, abs r)
	signum (Chain (l, r)) = Chain (signum l, signum r)
	fromInteger i = Chain (fromInteger i, fromInteger i)
	negate (Chain (l, r)) = Chain (negate l, negate r)

instance (Fractional l, Fractional r) ⇒ Fractional (Chain l r) where
	fromRational r = Chain (fromRational r, fromRational r)
	recip (Chain (l, r)) = Chain (recip l, recip r)

newtype Named a = Named { getNamed ∷ (String, a) } deriving (Eq, Ord, Read, Functor, Applicative)

instance Show a ⇒ Show (Named a) where
	show (Named (n, v)) = unlines ["name: " ++ n, show v]

instance Num a ⇒ Num (Named a) where
	(+) = liftA2 (+)
	(*) = liftA2 (*)
	abs = liftA abs
	signum = liftA signum
	fromInteger i = Named ("", fromInteger i)
	negate = liftA negate

instance Fractional a ⇒ Fractional (Named a) where
	fromRational r = Named ("", fromRational r)
	recip = liftA recip

type Parametric w = (Read w, Show w, Num w, Fractional w, LookupParams w)

class LookupParams a where
	lookupParams ∷ String → a → (forall w . Parametric w ⇒ w → r) → r → r
	traverseParams ∷ Traversal' a String

instance LookupParams NoParams where
	lookupParams _ _ _ act = act
	traverseParams _ = pure

instance Parametric a ⇒ LookupParams (Named a) where
	lookupParams n (Named (n', v)) fn def
		| n ≡ n' = fn v
		| otherwise = def
	traverseParams f (Named (n, v)) = Named <$> ((,) <$> f n <*> pure v)

instance (Parametric a, Parametric b) ⇒ LookupParams (Chain a b) where
	lookupParams n (Chain (x, y)) fn def = lookupParams n x fn (lookupParams n y fn def)
	traverseParams f (Chain (x, y)) = Chain <$> ((,) <$> traverseParams f x <*> traverseParams f y)

instance {-# OVERLAPPABLE #-} LookupParams a where
	lookupParams _ _ _ def = def
	traverseParams _ = pure

data LearneeT w a b = LearneeT {
	_params ∷ w,
	_pass ∷ w → a → (b, b → (a, w)) }

makeLenses ''LearneeT

mapP ∷ Iso' w w' → LearneeT w a b → LearneeT w' a b
mapP fn (LearneeT ws f) = LearneeT (view fn ws) f' where
	f' ws' x = (y, A.second (view fn) ∘ back) where
		(y, back) = f (view (from fn) ws') x

name ∷ String → Learnee a b → Learnee a b
name n l = withLearnee l $ \l' → toLearnee (mapP (iso setName dropName) l') where
	setName ∷ w → Named w
	setName w = Named (n, w)
	dropName ∷ Named w → w
	dropName (Named (_, w)) = w

data Learnee a b where
	Learnee ∷ Parametric w ⇒ w → (w → a → (b, b → (a, w))) → Learnee a b

withLearnee ∷ Learnee a b → (forall w . Parametric w ⇒ LearneeT w a b → r) → r
withLearnee (Learnee ws f) fn = fn (LearneeT ws f)

withParams ∷ Learnee a b → (forall w .  Parametric w ⇒ w → r) → r
withParams (Learnee ws _) fn = fn ws

toLearnee ∷ Parametric w ⇒ LearneeT w a b → Learnee a b
toLearnee (LearneeT ws f) = Learnee ws f

instance LookupParams (Learnee a b) where
	lookupParams n (Learnee ws _) = lookupParams n ws
	traverseParams f (Learnee ws fn) = Learnee <$> traverseParams f ws <*> pure fn

instance Parametric w ⇒ LookupParams (LearneeT w a b) where
	lookupParams n (LearneeT ws _) = lookupParams n ws
	traverseParams f (LearneeT ws fn) = LearneeT <$> traverseParams f ws <*> pure fn

type Cost b = b → b → (b, b)

type Example a b = (a, b)
