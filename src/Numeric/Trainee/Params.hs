{-# LANGUAGE GADTs, RankNTypes, ConstraintKinds, TypeOperators, FlexibleInstances #-}

module Numeric.Trainee.Params (
	Parametric,
	Params(..),
	onParams, liftParams,
	onParams2, liftParams2
	) where

import Prelude.Unicode

import Control.DeepSeq
import Data.List (intercalate, intersperse)
import Data.Typeable
import qualified Data.Vector as V

-- | Constraints to smth, that can be parameters
type Parametric w = (Show w, Num w, Fractional w, NFData w, Typeable w)

-- | Parameters holder, we use it in @Learnee@ not to pass params type, as long as
-- combining many @Learnee@s will produce huge and unreadable params type
data Params where
	Params ∷ Parametric w ⇒ w → Params
	deriving (Typeable)

instance Show Params where
	show (Params ws) = show ws

instance Num Params where
	(+) = liftParams2 (+)
	(*) = liftParams2 (*)
	abs = liftParams abs
	signum = liftParams signum
	fromInteger = Params ∘ (fromInteger ∷ Integer → Double)
	negate = liftParams negate

instance Fractional Params where
	fromRational = Params ∘ (fromRational ∷ Rational → Double)
	recip = liftParams recip

instance NFData Params where
	rnf (Params ws) = rnf ws

instance {-# OVERLAPPING #-} Show (Params, Params) where
	show (l, r) = intercalate "\n" $ intersperse (replicate 10 '-') $
		filter (not ∘ null) [show l, show r]

instance Num (Params, Params) where
	(l, r) + (l', r') = (l + l', r + r')
	(l, r) * (l', r') = (l * l', r * r')
	abs (l, r) = (abs l, abs r)
	signum (l, r) = (signum l, signum r)
	fromInteger i = (fromInteger i, fromInteger i)
	negate (l, r) = (negate l, negate r)

instance Fractional (Params, Params) where
	fromRational r = (fromRational r, fromRational r)
	recip (l, r) = (recip l, recip r)

instance {-# OVERLAPPING #-} Show (V.Vector Params) where
	show ps = intercalate "\n" $ intersperse (replicate 10 '-') $
		filter (not ∘ null) ∘ map show ∘ V.toList $ ps

instance Num (V.Vector Params) where
	ls + rs = V.zipWith (+) ls rs
	ls * rs = V.zipWith (*) ls rs
	abs = V.map abs
	signum = V.map signum
	fromInteger = V.singleton ∘ fromInteger
	negate = V.map negate

instance Fractional (V.Vector Params) where
	fromRational = V.singleton ∘ fromRational
	recip = V.map recip

onParams ∷ (forall w . Parametric w ⇒ w → a) → Params → a
onParams fn (Params ws) = fn ws

liftParams ∷ (forall w . Parametric w ⇒ w → w) → Params → Params
liftParams fn = onParams (Params ∘ fn)

onParams2 ∷ (forall w . Parametric w ⇒ w → w → a) → Params → Params → a
onParams2 fn (Params lws) (Params rws) = case eqT' lws rws of
	Just Refl → fn lws rws
	_ → case (asDouble lws, asDouble rws) of
		(Just _, Just _) → error "onParams2: impossible, inequal types, but both doubles"
		(Just d, Nothing) → fn (fromRational (toRational d)) rws
		(Nothing, Just d) → fn lws (fromRational (toRational d))
		(Nothing, Nothing) → error $ "params type mismatch: '" ++ typeName lws ++ "' and '" ++ typeName rws ++ "'"
	where
		eqT' ∷ (Typeable u, Typeable v) ⇒ u → v → Maybe (u :~: v)
		eqT' _ _ = eqT
		asDouble ∷ Typeable a ⇒ a → Maybe Double
		asDouble = cast
		typeName ∷ Typeable a ⇒ a → String
		typeName = show ∘ typeRep ∘ proxy'
		proxy' ∷ a → Proxy a
		proxy' _ = Proxy

liftParams2 ∷ (forall w . Parametric w ⇒ w → w → w) → Params → Params → Params
liftParams2 fn = onParams2 ((Params ∘) ∘ fn)
